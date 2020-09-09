#!/usr/bin/env python
import os
import shutil

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class DeleteDirs(BasePipeline):
    """Delete dirs distributed on several workers pipeline."""

    def run(self):
        """Run prod."""
        target_prefix = self.FLAGS.get('input_data_path')
        if not self.sanity_check(target_prefix):
            logging.info('Absolutely beyond prediction!')
            return
        (
            # RDD(file), start with target_prefix
            self.to_rdd(self.our_storage().list_files(target_prefix))
            # remove everyfile
            .foreach(os.remove))

        shutil.rmtree(os.path.join('/mnt/bos/', target_prefix))
        logging.info('Delete dirs name: ' + target_prefix)
        logging.info('Nothing is beyond prediction!')

    def sanity_check(self, target):
        if not target:
            return False
        path_list = [path for path in target.split('/') if path != '']
        logging.info(f'target directory list: {path_list}')
        if ((len(path_list) == 1) or (len(path_list) == 3 and path_list[0] == 'mnt')):
            return False
        else:
            return True


if __name__ == '__main__':
    DeleteDirs().main()
