#!/usr/bin/env python
"""Clean records."""

import datetime
import os

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline


class EmptyFolder(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        pass

    def run_test(self):
        """Run test."""
        pass

    def run(self):
        """Run prod."""

        prefixes = [
            '/mnt/bos/modules/planning/cleaned_data/',
        ]

        self.to_rdd(prefixes).map(self.process_folder).count()

        logging.info('Processing is done')

    def process_folder(self, empty_folder):
        for filename in os.listdir(empty_folder):
            file_path = os.path.join(empty_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    pass

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        return 1


if __name__ == '__main__':
    cleaner = EmptyFolder()
    cleaner.main()
