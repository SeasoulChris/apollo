#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class TrajectoryEvaluator(BasePipeline):
    def __init__(self):
        self.src_dir_prefixs = [
            'modules/planning/output_data/test/',
        ]

    def run_test(self):
        """Run Test"""
        self.src_dir_prefixs = [
            '/fuel/data/output_data/test/',
        ]

        src_dirs_set = set([])
        for prefix in self.src_dir_prefixs:
            for root, dirs, files in os.walk(prefix):
                for file in files:
                    src_dirs_set.add(root)

        processed_dirs = self.to_rdd(src_dirs_set).map(self.process_dir)

        logging.info('Processed {}/{} folders'.format(processed_dirs.count(),
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

        processed_dirs = records_rdd.map(self.process_dir)

        logging.info('Processed {} folders'.format(processed_dirs.count()))

    def process_dir(self, src_dir):
        """ Process files """
        src_dir_elements = src_dir.split("/")
        # timestamp = [ i for i in src_dir_elements if i.startswith('ver_') ]
        dest_dir_elements = ['output_data_evaluated' if x ==
                             'output_data' else x for x in src_dir_elements]
        if ('output_data_evaluated' in dest_dir_elements):
            dest_dir = "/".join(dest_dir_elements)
        else:
            dest_dir = "/".join(src_dir_elements)

        file_utils.makedirs(dest_dir)

        """Call planning C++ code."""
        map_name = "sunnyvale_with_two_offices"
        command = (
            'cd /apollo && sudo bash '
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
