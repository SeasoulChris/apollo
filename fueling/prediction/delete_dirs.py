#!/usr/bin/env python
import operator
import os
import shutil

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class DeleteDirs(BasePipeline):
    """Delete dirs distributed on several workers pipeline."""

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
