#!/usr/bin/env python
import operator
import os
import shutil

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.record_utils as record_utils


class DeleteDirs(BasePipeline):
    """Records to feature proto pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'delete-dirs')

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.to_rdd(['/apollo/docs/demo_guide'])
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data/prediction/labels/'

    def run_prod(self):
        """Run prod."""
        target_prefix = 'modules/prediction/ground_truth/'
        files = (
            # RDD(file), start with target_prefix
            self.to_rdd(self.bos().list_files(target_prefix))
            # remove everyfile
            .foreach(os.remove))

        shutil.rmtree(os.path.join('/mnt/bos/', target_prefix))
        glog.info('Delete dirs name: ' + target_prefix)
        glog.info('Everything is under control!')


if __name__ == '__main__':
    DeleteDirs().main()
