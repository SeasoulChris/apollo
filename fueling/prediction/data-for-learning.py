#!/usr/bin/env python
import operator
import os

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class DataForLearning(BasePipeline):
    """Records to DataForLearning proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'data-for-learning')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        root_dir = '/apollo'
        # RDD(dir_path)
        records_dir = sc.parallelize(['docs/demo_guide'])
        origin_prefix = 'docs/demo_guide'
        target_prefix = 'data/prediction/features/'
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/'
        target_prefix = 'modules/prediction/features/'

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
            # PairRDD(record_dir, target_dir)
            .map(lambda record_dir: (record_dir,
                                     record_dir.replace(origin_prefix, target_prefix, 1)))
            # PairRDD(record_dir, target_dir), in absolute path
            .map(lambda src_dst: (os.path.join(root_dir, src_dst[0]),
                                  os.path.join(root_dir, src_dst[1])))
            # RDD(0/1), 1 for success
            .map(spark_op.do_tuple(self.process_dir))
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(src_dir, target_dir):
        """Call prediction C++ code to get data-for-learning."""
        # use /apollo/hmi/status's current_map entry to match map info
        map_dir = record_utils.get_map_name_from_records(src_dir)
        target_dir = os.path.join(target_dir, map_dir)
        command = (
            'cd /apollo && '
            'bash modules/tools/prediction/data_pipelines/scripts/records_to_data_for_learning.sh '
            '"{}" "{}" "{}"'.format(src_dir, target_dir, map_dir))
        if os.system(command) == 0:
            glog.info('Successfuly processed {} to {}'.format(src_dir, target_dir))
            return 1
        else:
            glog.error('Failed to process {} to {}'.format(src_dir, target_dir))
        return 0


if __name__ == '__main__':
    DataForLearning().run_prod()
