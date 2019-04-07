#!/usr/bin/env python
import collections
import os
import sys

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
from fueling.data.record_parser import RecordParser
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class IndexRecords(BasePipeline):
    """IndexRecords pipeline."""
    COLLECTION_NAME = 'records'

    def __init__(self):
        BasePipeline.__init__(self, 'index-records')

    def run_test(self):
        """Run test."""
        self.process(
            # RDD(record_path)
            self.get_spark_context().parallelize(['/apollo/docs/demo_guide/demo_3.5.record']))

    def run_prod(self):
        """Run prod."""
        summary_receivers = ['apollo_internal@baidu.com', 'xiaoxiangquan@baidu.com']
        bucket = 'apollo-platform'
        prefix = 'public-test/'
        records_rdd = (
            # RDD(file_path)
            s3_utils.list_files(bucket, prefix)
            # RDD(record_path)
            .filter(record_utils.is_record_file)
            # RDD(record_path), with absolute path.
            .map(s3_utils.abs_path))
        self.process(records_rdd, summary_receivers)

    def process(self, records_rdd, summary_receivers=None):
        """Run the pipeline with given arguments."""
        collection = Mongo.collection(self.COLLECTION_NAME)
        docs = collection.find({}, {'path': 1})
        imported_records = [doc['path'] for doc in docs]
        glog.info('Found {} imported records'.format(len(imported_records)))

        newly_imported_records = spark_op.log_rdd(
            records_rdd
            # RDD(record_path), which is not imported before.
            .subtract(self.get_spark_context().parallelize(imported_records))
            # RDD(RecordMeta)
            .map(RecordParser.Parse)
            # RDD(RecordMeta), which is valid.
            .filter(lambda record_meta: record_meta is not None)
            # RDD(RecordMeta_doc)
            .map(Mongo.pb_to_doc)
            # RDD(imported_path)
            .mapPartitions(self.import_record),
            "NewlyImportedRecords", glog.info)
        if summary_receivers:
            self.send_summary(newly_imported_records, summary_receivers)

    @staticmethod
    def import_record(record_meta_docs):
        """Import record docs to Mongo."""
        collection = Mongo.collection(IndexRecords.COLLECTION_NAME)
        newly_imported = []
        for doc in record_meta_docs:
            collection.replace_one({'path': doc['path']}, doc, upsert=True)
            newly_imported.append(doc['path'])
            glog.info('Imported record {}'.format(doc['path']))
        return newly_imported

    @staticmethod
    def send_summary(imported_records_rdd, receivers):
        """Send summary."""
        SummaryTuple = collections.namedtuple('Summary', ['Path', 'URL'])

        proxy = 'http://usa-data.baidu.com:8001'
        service = 'http:warehouse-service:8000'
        url_prefix = '{}/api/v1/namespaces/default/services/{}/proxy/task'.format(proxy, service)

        msgs = (imported_records_rdd
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
        else:
            title = 'Imported {} tasks into Apollo Fuel Warehouse'.format(msg_count)
            email_utils.send_email_info(title, msgs.collect(), receivers)


if __name__ == '__main__':
    IndexRecords().main()
