#!/usr/bin/env python
import collections
import os
import sys

import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
from fueling.data.record_parser import RecordParser
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.db_backed_utils as db_backed_utils


class IndexRecords(BasePipeline):
    """IndexRecords pipeline."""
    COLLECTION_NAME = 'records'

    def __init__(self):
        BasePipeline.__init__(self, 'index-records')

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        self.process(self.context().parallelize(['/apollo/docs/demo_guide/demo_3.5.record']))

    def run_prod(self):
        """Run prod."""
        summary_receivers = ['apollo_internal@baidu.com', 'xiaoxiangquan@baidu.com']
        bucket = 'apollo-platform'
        prefixes = [
            'public-test/',
            'small-records/',
        ]
        # RDD(record_path)
        records_rdd = self.context().union([
            s3_utils.list_files(bucket, prefix).filter(record_utils.is_record_file)
            for prefix in prefixes])
        self.process(records_rdd, summary_receivers)

    def process(self, records_rdd, summary_receivers=None):
        """Run the pipeline with given arguments."""
        docs = Mongo.collection(self.COLLECTION_NAME).find({}, {'path': 1})
        # RDD(record_path), which is indexed before.
        indexed_records = spark_helper.cache_and_log(
            'IndexedRecords', self.context().parallelize([doc['path'] for doc in docs]))
        # RDD(record_path), which is not indexed.
        records_rdd = spark_helper.cache_and_log(
            'RecordsToIndex', records_rdd.subtract(indexed_records))
        # RDD(record_path), which is newly indexed.
        new_indexed_records = spark_helper.cache_and_log(
            'NewlyIndexedRecords', records_rdd.mapPartitions(self.index_records))
        if summary_receivers:
            self.send_summary(new_indexed_records, summary_receivers)

    @staticmethod
    def index_records(records):
        """Import record docs to Mongo."""
        records = list(records)
        collection = Mongo.collection(self.COLLECTION_NAME)
        indexed_records = set(db_backed_utils.lookup_existing_records(records, collection))
        new_indexed = []
        for record in records:
            if record in indexed_records:
                new_indexed.append(record)
                glog.info('Skip record indexed in current batch: {}'.format(record))
                continue
            record_meta = RecordParser.Parse(record)
            if record_meta is None:
                continue
            doc = Mongo.pb_to_doc(record_meta)
            collection.replace_one({'path': doc['path']}, doc, upsert=True)
            new_indexed.append(record)
            glog.info('Indexed record {}'.format(record))
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
            glog.error('No record was imported')
            return

        title = 'Imported {} tasks into Apollo Fuel Warehouse'.format(msg_count)
        try:
            email_utils.send_email_info(title, msgs.collect(), receivers)
        except Exception as error:
            glog.error('Failed to send summary: {}'.format(error))


if __name__ == '__main__':
    IndexRecords().main()
