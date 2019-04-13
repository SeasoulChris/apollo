#!/usr/bin/env python
import operator
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class PredictionResult(BasePipeline):
    """Records to PredictionResult proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'prediction-result')

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.context().parallelize(['/apollo/docs/demo_guide'])
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data/prediction/results'
        self.run(records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/'
        target_prefix = 'modules/prediction/results/'
        records_dir = (
            # RDD(file), start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())
        self.run(records_dir, origin_prefix, target_prefix)

    def run(self, records_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            records_dir_rdd
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
        target_dir = record_dir.replace(origin_prefix, os.path.join(target_prefix, map_dir))
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_prediction_result.sh '
            '"{}" "{}" "{}"'.format(record_dir, target_dir, map_dir))
        if os.system(command) == 0:
            glog.info('Successfuly processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            glog.error('Failed to process {} to {}'.format(record_dir, target_dir))
        return 0


if __name__ == '__main__':
    PredictionResult().run_prod()
