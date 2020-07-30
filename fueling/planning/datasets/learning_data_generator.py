#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class LearningDataGenerator(BasePipeline):
    """Records to feature proto pipeline."""

    def __init__(self):
        self.src_dir_prefixs = [
            'modules/planning/cleaned_data/',
        ]

    def run(self):
        if self.is_local():
            self.src_dir_prefixs = [
                '/fuel/data/cleaned_data/',
            ]

        for prefix in self.src_dir_prefixs:
            self.run_internal(prefix)

    def run_internal(self, src_dir_prefix):
        data_dir_rdd = (
            self.to_rdd(self.our_storage().list_files(src_dir_prefix))
                .filter(record_utils.is_record_file)
                .map(os.path.dirname)
                .distinct())

        processed_dirs = data_dir_rdd.map(
            lambda src_dir: self.process_dir(src_dir_prefix, src_dir))

        logging.info('Processed {} folders'.format(processed_dirs.count()))

    def process_dir(self, src_dir_prefix, src_dir):
        """ Process Records """
        src_dir_elements = src_dir.split("/")
        # timestamp = [ i for i in src_dir_elements if i.startswith('ver_') ]
        dest_dir_elements = ['learning_data' if x
                             == 'cleaned_data' else x for x in src_dir_elements]
        if ('learning_data' in dest_dir_elements):
            dest_dir = "/".join(dest_dir_elements)
        else:
            dest_dir_elements = src_dir_prefix.split("/")
            while (dest_dir_elements[-1] == ''):
                dest_dir_elements.pop()
            dest_dir_elements[-1] += '_learning_data'
            prefix_len = len(dest_dir_elements)
            dest_dir_elements.extend(src_dir_elements[prefix_len:])
            dest_dir = "/".join(dest_dir_elements)

        map_name = "sunnyvale_with_two_offices"
        if ('san_mateo' in dest_dir_elements):
            map_name = "san_mateo"

        file_utils.makedirs(dest_dir)

        """Call planning C++ code."""
        command = (
            'cd /apollo && bash '
            'modules/tools/planning/data_pipelines/scripts/'
            'record_to_learning_data.sh '
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
    LearningDataGenerator().main()
