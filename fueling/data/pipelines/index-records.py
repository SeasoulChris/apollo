#!/usr/bin/env python
import os
import sys

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
from fueling.data.record_parser import RecordParser
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class IndexRecords(BasePipeline):
    """IndexRecords pipeline."""
    COLLECTION_NAME = 'records'

    def __init__(self):
        BasePipeline.__init__(self, 'index-records')
        self.indexed_records_acc = self.get_spark_context().accumulator(0)

    def run_test(self):
        """Run test."""
        self.process(
            # RDD(record_path)
            self.get_spark_context().parallelize(['/apollo/docs/demo_guide/demo_3.5.record']))

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        prefixes = [
            'small-records/2018/2018-04-02/',
        ]
        self.process(
            # RDD(file_path)
            self.get_spark_context().union(
                *[s3_utils.list_files(bucket, prefix) for prefix in prefixes])
            # RDD(record_path)
            .filter(record_utils.is_record_file)
            # RDD(record_path), with absolute path.
            .map(lambda record_path: os.path.join(s3_utils.S3_MOUNT_PATH, record_path)))

    def process(self, records_rdd):
        """Run the pipeline with given arguments."""
        collection = Mongo.collection(self.COLLECTION_NAME)
        docs = collection.find({}, {'path': 1})
        imported_records = [doc['path'] for doc in docs]
        glog.info('Found {} imported records'.format(len(imported_records)))

        (records_rdd
            # RDD(record_path), which is not imported before.
            .subtract(self.get_spark_context().parallelize(imported_records))
            # RDD(RecordMeta)
            .map(RecordParser.Parse)
            # RDD(RecordMeta), which is valid.
            .filter(lambda record_meta: record_meta is not None)
            # RDD(RecordMeta_doc)
            .map(Mongo.pb_to_doc)
            # RDD(imported_path)
            .mapPartitions(self.import_record))
        glog.info('Imported {} records'.format(self.indexed_records_acc.value))

    def import_record(self, record_meta_docs):
        """Import record docs to Mongo."""
        collection = Mongo.collection(self.COLLECTION_NAME)
        newly_imported = []
        for doc in record_meta_docs:
            collection.replace_one({'path': doc['path']}, doc, upsert=True)
            newly_imported.append(doc['path'])
            self.indexed_records_acc += 1
            glog.info('Imported record {}'.format(doc['path']))
        return newly_imported


if __name__ == '__main__':
    IndexRecords().run_prod()
