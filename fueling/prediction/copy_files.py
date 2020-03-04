#!/usr/bin/env python
import operator
import os
import shutil

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class CopyFiles(BasePipeline):
    """Records to feature proto pipeline."""
    def run(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/kinglong/'
        target_prefix = 'modules/prediction/test/'

        files_rdd = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(origin_prefix))
            # RDD(file), which is unique
            .distinct()
            .cache())

        dirs_rdd = (
            # RDD(file)
            files_rdd
            # RDD(target_file)
            .map(lambda path: path.replace(origin_prefix, target_prefix))
            # RDD(target_dir), with target file inside
            .map(os.path.dirname)
            # RDD(target_dir), which is unique
            .distinct()
            .foreach(file_utils.makedirs))

        result = files_rdd.map(lambda filename: shutil.copyfile(
                               filename, filename.replace(origin_prefix, target_prefix)))

        logging.info('Finishing copy ' + str(result.count()) + ' files: '
                     + origin_prefix + ' -> ' + target_prefix)
        logging.info('Everything is under control!')


if __name__ == '__main__':
    CopyFiles().main()
