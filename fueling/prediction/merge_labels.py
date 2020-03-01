#!/usr/bin/env python
import glob
import operator
import os

import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


SKIP_EXISTING_DST_FILE = False


class MergeLabels(BasePipeline):
    """Records to MergeLabels proto pipeline."""

    def run_test(self):
        """Run test."""
        # RDD(npy_file)
        npy_file_rdd = self.to_rdd(glob.glob('/apollo/data/prediction/labels/*/*.npy'))
        self.run_internal(npy_file_rdd.map(os.path.dirname))

    def run(self):
        """Run prod."""
        source_prefix = 'modules/prediction/labels/'

        npy_dirs = (
            # RDD(npy_files)
            self.to_rdd(self.our_storage().list_files(source_prefix, '.npy'))
            # RDD(target_dir), in absolute path
            .map(os.path.dirname)
            # RDD(target_dir), in absolute path and unique
            .distinct())

        merged_dirs = (
            # RDD(merged_label_files)
            self.to_rdd(self.our_storage().list_files(source_prefix, '/future_status.npy'))
            # RDD(target_dir), in absolute path
            .map(os.path.dirname)
            # RDD(target_dir), in absolute path and unique
            .distinct())

        # RDD(todo_npy_dirs)
        todo_npy_dirs = npy_dirs

        if SKIP_EXISTING_DST_FILE:
            # RDD(todo_npy_dirs)
            todo_npy_dirs = todo_npy_dirs.subtract(merged_dirs).distinct()

        self.run_internal(todo_npy_dirs)

    def run_internal(self, npy_dir_rdd):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(npy_file)
            npy_dir_rdd
            # RDD(target_dir), in absolute path and unique
            .distinct()
            # RDD(0/1), 1 for success
            .map(self.process_dir)
            .cache())

        if result.isEmpty():
            logging.info("Nothing to be processed, everything is under control!")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(src_dir):
        """Call prediction python code to merge labels."""
        try:
            merge_dicts(src_dir, dict_name='future_status')
            merge_dicts(src_dir, dict_name='junction_label')
            merge_dicts(src_dir, dict_name='cruise_label')
            merge_dicts(src_dir, dict_name='visited_lane_segment')
            logging.info('Successfully processed {}'.format(src_dir))
            return 1
        except BaseException:
            logging.error('Failed to process {}'.format(src_dir))
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
    MergeLabels().main()
