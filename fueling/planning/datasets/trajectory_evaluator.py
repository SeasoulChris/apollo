#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.context_utils as context_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class TrajectoryEvaluator(BasePipeline):
    def __init__(self):
        self.src_dir_prefixs = [
            'modules/planning/output_data/',
        ]

    def run(self):
        if context_utils.is_local():
            self.src_dir_prefixs = [
                '/fuel/data/output_data',
            ]

        for prefix in self.src_dir_prefixs:
            self.run_internal(prefix)

    def run_internal(self, src_dir_prefix):
        data_dir_rdd = (
            self.to_rdd(self.our_storage().list_files(src_dir_prefix))
                .map(os.path.dirname)
                .distinct())

        processed_dirs = data_dir_rdd.map(
            lambda src_dir: self.process_dir(src_dir_prefix, src_dir))

        logging.info('Processed {} folders'.format(processed_dirs.count()))

    def process_dir(self, src_dir_prefix, src_dir):
        """ Process files """
        src_dir_elements = src_dir.split("/")
        # timestamp = [ i for i in src_dir_elements if i.startswith('ver_') ]
        dest_dir_elements = ['output_data_evaluated' if x
                             == 'output_data' else x for x in src_dir_elements]
        if ('output_data_evaluated' in dest_dir_elements):
            dest_dir = "/".join(dest_dir_elements)
        else:
            dest_dir_elements = src_dir_prefix.split("/")
            while (dest_dir_elements[-1] == ''):
                dest_dir_elements.pop()
            dest_dir_elements[-1] += '_output_data_evaluated'
            prefix_len = len(dest_dir_elements)
            dest_dir_elements.extend(src_dir_elements[prefix_len:])
            dest_dir = "/".join(dest_dir_elements)

        file_utils.makedirs(dest_dir)

        """Call planning C++ code."""
        command = (
            'cd /apollo && bash '
            'modules/tools/planning/data_pipelines/scripts/'
            'evaluate_trajectory.sh '
            '"{}" "{}"'.format(src_dir, dest_dir))

        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(src_dir,
                                                                  dest_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(src_dir,
                                                              dest_dir))

        return 0


if __name__ == '__main__':
    TrajectoryEvaluator().main()
