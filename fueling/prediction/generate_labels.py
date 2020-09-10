#!/usr/bin/env python
import os
import operator

from fueling.common.base_pipeline import BasePipeline
from fueling.prediction.common.online_to_offline import LabelGenerator
import fueling.common.context_utils as context_utils
from fueling.common.job_utils import JobUtils
import fueling.common.logging as logging
import fueling.common.spark_op as spark_op


SKIP_EXISTING_DST_FILE = False


class GenerateLabels(BasePipeline):
    """Records to GenerateLabels proto pipeline."""
    def __init__(self):
        super(GenerateLabels, self).__init__()
        self.is_on_cloud = context_utils.is_cloud()
        self.if_error = False

    def run(self):
        # input_path = 'modules/prediction/kinglong_benchmark'
        input_path = self.FLAGS.get('input_path')
        self.source_prefix = os.path.join(input_path, 'labels')
        # RDD(bin_files)
        bin_files = (
            self.to_rdd(self.our_storage().list_files(self.source_prefix)).filter(
                spark_op.filter_path(['*feature.*.bin'])))
        labeled_bin_files = (
            # RDD(label_files)
            self.to_rdd(self.our_storage().list_files(self.source_prefix, '.bin.future_status.npy'))
            # RDD(bin_files)
            .map(lambda label_file: label_file.replace('.bin.future_status.npy', '.bin')))
        # RDD(todo_bin_files)
        todo_bin_files = bin_files

        if SKIP_EXISTING_DST_FILE:
            # RDD(todo_bin_files)
            todo_bin_files = todo_bin_files.subtract(labeled_bin_files).distinct()

        self.run_internal(todo_bin_files)

        if self.is_on_cloud:
            job_id = (self.FLAGS.get('job_id') if self.is_partner_job() else
                      self.FLAGS.get('job_id')[:4])
            JobUtils(job_id).save_job_progress(20)
            if self.if_error:
                error_text = 'Failed to generate labels.'
                JobUtils(job_id).save_job_failure_code('E603')
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     error_text, False)

    def run_internal(self, bin_files_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = bin_files_rdd.map(self.process_file).cache()

        if result.isEmpty():
            logging.info("Nothing to be processed, everything is under control!")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    def process_file(self, src_file):
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
            self.if_error = True
        return 0


if __name__ == '__main__':
    GenerateLabels().main()
