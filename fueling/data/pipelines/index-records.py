#!/usr/bin/env python
import os

import gflags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
from fueling.data.record_parser import RecordParser
import fueling.common.colored_glog as glog
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils


gflags.DEFINE_string('mongo_collection_name', 'records', 'MongoDB collection name.')


class IndexRecords(BasePipeline):
    """IndexRecords pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'index-records')
        self.mongo_collection = None

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
            self.get_spark_context().union(*[
                s3_utils.list_files(bucket, prefix) for prefix in prefixes])
            # RDD(record_path)
            .filter(record_utils.is_record_file)
            # RDD(record_path), with absolute path.
            .map(lambda record_path: os.path.join(s3_utils.S3_MOUNT_PATH, record_path)))

    def process(self, records_rdd):
        """Run the pipeline with given arguments."""
        docs = self._get_mongo_collection().find({}, {'path': 1})
        imported_records = [doc['path'] for doc in docs]
        glog.info('Found {} imported records'.format(len(imported_records)))

        result = (records_rdd
            # RDD(record_path), which is not imported before.
            .subtract(self.get_spark_context().parallelize(imported_records))
            # RDD(RecordMeta)
            .map(RecordParser.Parse)
            # RDD(RecordMeta), which is valid.
            .filter(lambda record_meta: record_meta is not None)
            # RDD(RecordMeta_doc)
            .map(Mongo.pb_to_doc)
            # RDD(None)
            .map(self.import_record)
            # Count.
            .count())
        glog.info('Imported {} records'.format(result))

    def import_record(self, record_meta_doc):
        """Import a record doc to Mongo."""
        self._get_mongo_collection().replace_one({'path': record_meta_doc['path']},
                                                 record_meta_doc, upsert=True)
        glog.info('Imported record {}'.format(path))

    def _get_mongo_collection(self):
        """Get a connected Mongo collection."""
        if self.mongo_collection is None:
            self.mongo_collection = Mongo.collection(gflags.FLAGS.mongo_collection_name)
        return self.mongo_collection


if __name__ == '__main__':
    gflags.FLAGS(sys.argv)
    IndexRecords().run_test()
