#!/usr/bin/env python
import os
import glob
import operator

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.prediction.common.online_to_offline import LabelGenerator
import fueling.common.logging as logging


SKIP_EXISTING_DST_FILE = False


class GenerateLabels(BasePipeline):
    """Records to GenerateLabels proto pipeline."""

    def run(self):
        """Run prod."""
        source_prefix = '/fuel/kinglong_data/labels/'
        if self.FLAGS.get('running_mode') == 'PROD':
            source_prefix = 'modules/prediction/kinglong_labels/'

        # RDD(bin_files)
        bin_files = (
            self.to_rdd(self.our_storage().list_files(source_prefix)).filter(
                spark_op.filter_path(['*feature.*.bin'])))
        labeled_bin_files = (
            # RDD(label_files)
            self.to_rdd(self.our_storage().list_files(source_prefix, '.bin.future_status.npy'))
            # RDD(bin_files)
            .map(lambda label_file: label_file.replace('.bin.future_status.npy', '.bin')))
        # RDD(todo_bin_files)
        todo_bin_files = bin_files

        if SKIP_EXISTING_DST_FILE:
            # RDD(todo_bin_files)
            todo_bin_files = todo_bin_files.subtract(labeled_bin_files).distinct()

        self.run_internal(todo_bin_files)

    def run_internal(self, bin_files_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = bin_files_rdd.map(self.process_file).cache()

        if result.isEmpty():
            logging.info("Nothing to be processed, everything is under control!")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_file(src_file):
        """Call prediction python code to generate labels."""
        label_gen = LabelGenerator()
        try:
            dir_name = os.path.dirname(src_file)
            command = "sudo chmod 777 {}".format(dir_name)
            os.system(command)
            label_gen.LoadFeaturePBAndSaveLabelFiles(src_file)
            label_gen.Label()
            logging.info('Successfully labeled {}'.format(src_file))
            return 1
        except BaseException:
            logging.error('Failed to process {}'.format(src_file))
        return 0


if __name__ == '__main__':
    GenerateLabels().main()
