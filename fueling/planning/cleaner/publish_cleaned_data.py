#!/usr/bin/env python
"""Clean records."""

import os
import shutil

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline


class PubishCleanedData(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        pass

    def run_test(self):
        """Run test."""
        pass

    def run(self):
        """Run prod."""

        prefixes = [
            '/mnt/bos/modules/planning/cleaned_data_temp/',
        ]

        self.to_rdd(prefixes).map(self.process_folder).count()

        logging.info('Processing is done')

    def process_folder(self, temp_cleaned_data):
        target_folder = '/mnt/bos/modules/planning/cleaned_data/'
        try:
            shutil.rmtree(target_folder)
        except:
            pass
        os.rename(temp_cleaned_data, target_folder)
        return 1


if __name__ == '__main__':
    cleaner = PubishCleanedData()
    cleaner.main()
