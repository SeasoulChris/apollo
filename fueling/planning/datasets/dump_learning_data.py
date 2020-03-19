#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class DumpLearningData(BasePipeline):
    """Records to feature proto pipeline."""

    def __init__(self):
        self.src_dir_prefixs = [
            'modules/planning/cleaned_data/',
        ]
        self.dest_dir_prefix = 'modules/planning/learning_data/'

    def run_test(self):
        """Run Test"""
        self.src_dir_prefixs = [
            '/apollo/data/cleaned_data/',
        ]
        self.dest_dir_prefix = '/apollo/data/learning_data/'

        src_dirs_set = set([])
        for prefix in self.src_dir_prefixs:
            for root, dirs, files in os.walk(prefix):
                for file in files:
                    src_dirs_set.add(root)

        processed_records = self.to_rdd(src_dirs_set).map(self.process_record)

        logging.info('Processed {}/{} records'.format(processed_records.count(),
                                                      len(src_dirs_set)))
        return 0

    def run(self):
        """Run"""
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix))
                .filter(record_utils.is_record_file)
                .map(os.path.dirname)
                .distinct()
            for prefix in self.src_dir_prefixs])

        processed_records = records_rdd.map(self.process_record)

        logging.info('Processed {} records'.format(processed_records.count()))

    def process_record(self, src_dir):
        """ Process Records """
        src_dir_elements = src_dir.split("/")
        # timestamp = [ i for i in src_dir_elements if i.startswith('ver_') ]
        dest_dir_elements = ['learning_data' if x ==
                             'cleaned_data' else x for x in src_dir_elements]
        if ('learning_data' in dest_dir_elements):
            dest_dir = "/".join(dest_dir_elements)
        else:
            dest_dir = "/".join(src_dir_elements)

        file_utils.makedirs(dest_dir)

        """Call planning C++ code."""
        map_name = "sunnyvale_with_two_offices"
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/planning/data_pipelines/scripts/'
            'records_to_data_for_learning.sh '
            '"{}" "{}" "{}"'.format(src_dir, dest_dir, map_name))

        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(src_dir,
                                                                  dest_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(src_dir,
                                                              dest_dir))

        return 0


if __name__ == '__main__':
    DumpLearningData().main()
