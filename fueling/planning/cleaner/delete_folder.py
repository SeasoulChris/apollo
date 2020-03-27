#!/usr/bin/env python
"""Clean records."""

import shutil
from absl import flags

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline

flags.DEFINE_string('to_be_deleted_folder',
                    '/mnt/bos/modules/planning/cleaned_data/ver_20200219_213417/',
                    'The folder to be deleted.')


class DeleteFolder(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        pass

    def run_test(self):
        """Run test."""
        pass

    def run(self):
        """Run prod."""
        folder = flags.FLAGS.to_be_deleted_folder
        logging.info(folder)

        prefixes = [folder]

        self.to_rdd(prefixes).map(self.process_folder).count()

        logging.info('Processing is done')

    def process_folder(self, empty_folder):
        root_folder = "/mnt/bos/modules/planning/"
        if empty_folder[0:len(root_folder)] != root_folder:
            return 1
        if len(empty_folder) <= len(root_folder):
            return 1
        shutil.rmtree(empty_folder)
        return 1


if __name__ == '__main__':
    cleaner = DeleteFolder()
    cleaner.main()
