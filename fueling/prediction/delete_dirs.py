#!/usr/bin/env python
import operator
import os
import shutil

from fueling.common.base_pipeline import BasePipeline
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class DeleteDirs(BasePipeline):
    """Records to feature proto pipeline."""

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.to_rdd(['/apollo/docs/demo_guide'])
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data/prediction/labels/'

    def run(self):
        """Run prod."""
        target_prefix = 'modules/prediction/ground_truth/'
        files = (
            # RDD(file), start with target_prefix
            self.to_rdd(self.our_storage().list_files(target_prefix))
            # remove everyfile
            .foreach(os.remove))

        shutil.rmtree(os.path.join('/mnt/bos/', target_prefix))
        logging.info('Delete dirs name: ' + target_prefix)
        logging.info('Everything is under control!')


if __name__ == '__main__':
    DeleteDirs().main()
