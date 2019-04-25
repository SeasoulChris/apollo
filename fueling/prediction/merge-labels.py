#!/usr/bin/env python
import glob
import operator
import os

import colored_glog as glog
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils


class MergeLabels(BasePipeline):
    """Records to MergeLabels proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'merge-labels')

    def run_test(self):
        """Run test."""
        # RDD(npy_file)
        npy_file_rdd = self.to_rdd(glob.glob('/apollo/data/prediction/labels/*/*.npy'))
        self.run(npy_file_rdd)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        source_prefix = 'modules/prediction/ground_truth'
        # RDD(npy_file)
        npy_file_rdd = s3_utils.list_files(bucket, source_prefix, '.npy')
        self.run(npy_file_rdd)

    def run(self, npy_file_rdd):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(npy_file)
            npy_file_rdd
            # RDD(target_dir), in absolute path
            .map(os.path.dirname)
            # RDD(target_dir), in absolute path and unique
            .distinct()
            # RDD(0/1), 1 for success
            .map(self.process_dir)
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(src_dir):
        """Call prediction python code to merge labels."""
        try:
            merge_dicts(src_dir, dict_name='future_status')
            merge_dicts(src_dir, dict_name='junction_label')
            merge_dicts(src_dir, dict_name='cruise_label')
            glog.info('Successfuly processed {}'.format(src_dir))
            return 1
        except:
            glog.error('Failed to process {}'.format(src_dir))
        return 0


def merge_dicts(dirpath, dict_name='future_status'):
    '''
    Merge all dictionaries directly under a directory
    '''
    list_of_files = os.listdir(dirpath)
    dict_merged = dict()
    for filename in list_of_files:
        full_path = os.path.join(dirpath, filename)
        if filename.endswith(dict_name + '.npy'):
            dict_curr = np.load(full_path).item()
            dict_merged.update(dict_curr)
    np.save(os.path.join(dirpath, dict_name + '.npy'), dict_merged)
    return dict_merged

if __name__ == '__main__':
    MergeLabels().run_prod()
