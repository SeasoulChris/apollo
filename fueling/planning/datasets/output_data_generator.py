#!/usr/bin/env python
import operator

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.planning.datasets.label_generator import LabelGenerator
import fueling.common.logging as logging

SKIP_EXISTING_DST_FILE = False


class OutputDataGenerator(BasePipeline):
    """output data for offline training"""

    def __init__(self):
        self.src_dir_prefix = 'modules/planning/learning_data'
        self.dest_dir_prefix = 'modules/planning/output_data'

    def run(self):
        logging.info(self.our_storage().list_files(self.src_dir_prefix))
        # RDD(bin_files)
        bin_files = (
            self.to_rdd(self.our_storage().list_files(self.src_dir_prefix))
            .filter(spark_op.filter_path(['*.bin'])))
        # labeled_bin_files = (
        #     # RDD(label_files)
        #     self.to_rdd(file_utils.list_files(src_dir_prefix))
        #     # RDD(bin_files)
        #     .map(lambda label_file: label_file.replace('.bin.future_status.npy', '.bin')))
        # RDD(todo_bin_files)
        todo_bin_files = bin_files
        logging.info(bin_files.collect())

        # if SKIP_EXISTING_DST_FILE:
        #     # RDD(todo_bin_files)
        #     todo_bin_files = todo_bin_files.subtract(labeled_bin_files).distinct()
        self.run_internal(todo_bin_files)

    def run_internal(self, bin_files_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = bin_files_rdd.map(self.process_file).cache()

        if result.isEmpty():
            logging.info("Nothing to be processed.")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_file(src_file):
        """Call prediction python code to generate labels."""
        label_gen = LabelGenerator()
        try:
            label_gen.LoadFeaturePBAndSaveLabelFiles(src_file)
            label_gen.Label()
            logging.info('Successfully labeled {}'.format(src_file))
            return 1
        except BaseException:
            logging.error('Failed to process {}'.format(src_file))
        return 0


if __name__ == '__main__':
    OutputDataGenerator().main()
