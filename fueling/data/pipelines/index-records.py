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


# Config.
SKIP_INDEXED_RECORD = True
# End of configs.


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
        prefix = 'public-test/'
        # RDD(record_path)
        records_rdd = s3_utils.list_files(bucket, prefix).filter(record_utils.is_record_file)
        self.process(records_rdd, summary_receivers)

    def process(self, records_rdd, summary_receivers=None):
        """Run the pipeline with given arguments."""
        if SKIP_INDEXED_RECORD:
            docs = Mongo.collection(self.COLLECTION_NAME).find({}, {'path': 1})
            indexed_records = [doc['path'] for doc in docs]
            glog.info('Found {} imported records'.format(len(indexed_records)))
            # RDD(record_path), which is not indexed before.
            records_rdd = records_rdd.subtract(self.context().parallelize(indexed_records))

        new_indexed_records = spark_helper.cache_and_log('NewlyImportedRecords',
            records_rdd
            # RDD(RecordMeta)
            .map(RecordParser.Parse)
            # RDD(RecordMeta), which is valid.
            .filter(spark_op.not_none)
            # RDD(RecordMeta_doc)
            .map(Mongo.pb_to_doc)
            # RDD(imported_path)
            .mapPartitions(self.import_records))
        if summary_receivers:
            self.send_summary(new_indexed_records, summary_receivers)

    @staticmethod
    def import_records(record_meta_docs):
        """Import record docs to Mongo."""
        collection = Mongo.collection(IndexRecords.COLLECTION_NAME)
        newly_imported = []
        for doc in record_meta_docs:
            collection.replace_one({'path': doc['path']}, doc, upsert=True)
            newly_imported.append(doc['path'])
            glog.info('Imported record {}'.format(doc['path']))
        return newly_imported

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
