#!/usr/bin/env python
import operator
import os
import shutil

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils

class CopyFiles(BasePipeline):
    """Records to feature proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'copy-files')

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.to_rdd(['/apollo/docs/demo_guide'])
        origin_prefix = '/apollo/docs/demo_guide/'
        target_prefix = '/apollo/data/prediction/test/'


    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/results/'
        target_prefix = 'modules/prediction/test/'

        files_rdd = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.bos().list_files(origin_prefix))
            # RDD(file), which is unique
            .distinct()
            .cache())

        dirs_rdd = (
            # RDD(file)
            files_rdd
            # RDD(file_dir), with file inside
            .map(os.path.dirname)
            # RDD(target_dir)
            .map(lambda path: path.replace(origin_prefix, target_prefix))
            # RDD(target_dir), which is unique
            .distinct()
            .foreach(file_utils.makedirs))

        result = files_rdd.map(lambda filename: shutil.copyfile(
                               filename, filename.replace(origin_prefix, target_prefix)))

        glog.info('Finishing copy ' + str(result.count()) + ' files: '
                  + origin_prefix + ' -> ' + target_prefix)
        glog.info('Everything is under control!')

if __name__ == '__main__':
    CopyFiles().main()
