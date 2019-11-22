#!/usr/bin/env python
"""Index records."""

import collections
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
from fueling.data.record_parser import RecordParser
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.common.redis_utils as redis_utils


PROCESS_LAST_N_DAYS = 7


class IndexRecords(BasePipeline):
    """IndexRecords pipeline."""

    def __init__(self):
        self.metrics_prefix = 'data.pipelines.index_records.'

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        records = ['/apollo/modules/data/fuel/testdata/data/small.record']
        indexed_records = (self.to_rdd(records)
                           # RDD(RecordMeta)
                           .map(RecordParser.Parse)
                           # RDD(RecordMeta), which is not None.
                           .filter(lambda meta: meta is not None)
                           .count())
        logging.info('Indexed {}/{} records'.format(indexed_records, len(records)))

    def run_prod(self):
        """Run prod."""
        prefixes = [
            'public-test/2019/',
            'small-records/2019/',
        ]
        bos = self.bos()
        # RDD(record_path)
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(bos.list_files(prefix)).filter(record_utils.is_record_file)
            for prefix in prefixes])

        # RDD(record_path), which is like /mnt/bos/small-records/2019/2019-09-09/...
        records_rdd = records_rdd.filter(
            record_utils.filter_last_n_days_records(PROCESS_LAST_N_DAYS))

        self.process(records_rdd, email_utils.DATA_TEAM)

    def process(self, records_rdd, summary_receivers=None):
        """Run the pipeline with given arguments."""
        docs = Mongo().record_collection().find({}, {'path': 1})
        indexed_records = [doc['path'] for doc in docs]
        redis_utils.redis_set(self.metrics_prefix + 'already_indexed_records', len(indexed_records))

        # RDD(record_path), which is not indexed.
        records_rdd = records_rdd.subtract(self.to_rdd(indexed_records)).cache()
        redis_utils.redis_set(self.metrics_prefix + 'records_to_be_indexed', records_rdd.count())
        redis_utils.redis_set(self.metrics_prefix + 'records_finished_indexing', 0)

        # RDD(record_path), which is newly indexed.
        new_indexed_records = records_rdd.mapPartitions(self.index_records)
        if summary_receivers:
            self.send_summary(new_indexed_records, summary_receivers)

    def index_records(self, records):
        """Import record docs to Mongo."""
        records = list(records)
        collection = Mongo().record_collection()
        indexed_records = set(db_backed_utils.lookup_existing_records(records, collection))
        new_indexed = []
        for record in records:
            if record in indexed_records:
                new_indexed.append(record)
                logging.info('Skip record indexed in current batch: {}'.format(record))
                continue
            record_meta = RecordParser.Parse(record)
            if record_meta is None:
                redis_utils.redis_incr(self.metrics_prefix + 'broken_records')
                continue
            doc = proto_utils.pb_to_dict(record_meta)
            collection.replace_one({'path': doc['path']}, doc, upsert=True)
            new_indexed.append(record)
            logging.info('Indexed record {}'.format(record))
            redis_utils.redis_incr(self.metrics_prefix + 'records_finished_indexing')
        return new_indexed

    @staticmethod
    def send_summary(new_indexed_records, receivers):
        """Send summary."""
        SummaryTuple = collections.namedtuple('Summary', ['Path', 'URL'])

        proxy = 'http://usa-data.baidu.com:8001'
        service = 'http:warehouse-service:8000'
        url_prefix = '{}/api/v1/namespaces/default/services/{}/proxy/task'.format(proxy, service)

        msgs = (new_indexed_records
                # RDD(imported_task_dir)
                .map(os.path.dirname)
                # RDD(imported_task_dir), which is unique
                .distinct()
                # RDD(SummaryTuple)
                .map(lambda task_dir: SummaryTuple(Path=task_dir, URL=url_prefix + task_dir))
                .cache())
        msg_count = msgs.count()
        if msg_count == 0:
            logging.error('No record was imported')
            return

        title = 'Index records: {}'.format(msg_count)
        try:
            email_utils.send_email_info(title, msgs.collect(), receivers)
        except Exception as error:
            logging.error('Failed to send summary: {}'.format(error))


if __name__ == '__main__':
    IndexRecords().main()
