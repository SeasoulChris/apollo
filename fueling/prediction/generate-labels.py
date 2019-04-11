#!/usr/bin/env python
import fnmatch
import glob
import operator
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from prediction.data_pipelines.common.online_to_offline import LabelGenerator

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class GenerateLabels(BasePipeline):
    """Records to GenerateLabels proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'generate-labels')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        # RDD(bin_files)
        bin_files = self.context().parallelize(
            glob.glob('/apollo/data/prediction/labels/*/feature.*.bin'))
        self.run(root_dir, bin_files)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        source_prefix = 'modules/prediction/labels/'

        bin_files  = (
            # RDD(file), start with source_prefix
            s3_utils.list_files(bucket, source_prefix)
            # RDD(bin_file)
            .filter(lambda src_file: fnmatch.fnmatch(src_file, '*feature.*.bin'))
            # RDD(record_dir), which is unique
            .distinct())
        self.run(root_dir, bin_files)

    def run(self, root_dir, bin_files_rdd):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(bin_files)
            bin_files_rdd
            # RDD(bin_files), in absolute path
            .map(lambda src_file: os.path.join(root_dir, src_file))
            # RDD(0/1), 1 for success
            .map(self.process_file)
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_file(src_file):
        """Call prediction python code to generate labels."""
        label_gen = LabelGenerator()
        try:
            label_gen.LoadFeaturePBAndSaveLabelFiles(src_file)
            label_gen.Label()
            glog.info('Successfuly labeled {}'.format(src_file))
            return 1
        except:
            glog.error('Failed to process {}'.format(src_file))
        return 0


if __name__ == '__main__':
    GenerateLabels().run_prod()
