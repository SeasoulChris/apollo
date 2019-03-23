#!/usr/bin/env python
import operator
import os

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class DumpFeatureProto(BasePipeline):
    """Records to feature proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'dump-feature-proto')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        root_dir = '/apollo'
        # RDD(dir_path)
        records_dir = sc.parallelize(['docs/demo_guide'])
        origin_prefix = 'docs/demo_guide'
        target_prefix = 'data/prediction/labels'
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/'
        target_prefix = 'modules/prediction/labels/'

        records_dir = (
            # RDD(file), start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run(self, root_dir, records_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            records_dir_rdd
            # RDD(record_dir), in absolute path
            .map(lambda record_dir: os.path.join(root_dir, record_dir))
            # PairRDD(record_dir, (origin_prefix, target_prefix))
            .map(lambda record_dir: (record_dir, (origin_prefix, target_prefix)))
            # RDD(0/1), 1 for success
            .map(self.process_dir)
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(input):
        """Call prediction C++ code."""
        record_dir, (origin_prefix, target_prefix) = input
        # use /apollo/hmi/status's current_map entry to match map info
        map_dir = record_utils.get_map_name_from_records(record_dir)
        target_dir = record_dir.replace(os.path.join(origin_prefix),
                                        os.path.join(target_prefix, map_dir))
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_dump_feature_proto.sh '
            '"{}" "{}" "{}"'.format(record_dir, target_dir, map_dir))
        if os.system(command) == 0:
            glog.info('Successfuly processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            glog.error('Failed to process {} to {}'.format(record_dir, target_dir))
        return 0


if __name__ == '__main__':
    DumpFeatureProto().run_prod()
