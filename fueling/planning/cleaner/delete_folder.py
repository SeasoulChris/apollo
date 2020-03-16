#!/usr/bin/env python
"""Clean records."""

import shutil

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline


class DeleteFolder(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        pass

    def run_test(self):
        """Run test."""
        pass

    def run(self):
        """Run prod."""

        prefixes = [
            '/mnt/bos/modules/planning/cleaned_data/ver_20200312_202938/',
        ]

        self.to_rdd(prefixes).map(self.process_folder).count()

        logging.info('Processing is done')

    def process_folder(self, empty_folder):
        shutil.rmtree(empty_folder)
        return 1


if __name__ == '__main__':
    cleaner = DeleteFolder()
    cleaner.main()
