#!/usr/bin/env python
import operator
import os

from fueling.common.base_pipeline_v2 import BasePipelineV2
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
from fueling.common.record.kinglong.cybertron.python.convert import convert_kinglong_to_apollo


SKIP_EXISTING_DST_FILE = True


class ConvertKinglongToApollo(BasePipelineV2):
    """Convert Kinglong record to Apollo pipeline."""

    def run(self):
        origin_prefix = 'kinglong/'
        target_prefix = 'modules/prediction/kinglong/'

        record_files = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(origin_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_file), which is unique
            .distinct())
        completed_record_files = (
            # RDD(output_file). start with target_prefix
            self.to_rdd(self.our_storage().list_files(target_prefix))
            # RDD(output_file)
            .filter(record_utils.is_record_file)
            # RDD(output_file), has been completed
            .map(lambda output_file: output_file.replace(target_prefix, origin_prefix))
            # RDD(output_file), which is unique
            .distinct())
        # RDD(todo_record_files)
        todo_record_files = record_files

        if SKIP_EXISTING_DST_FILE:
            # RDD(todo_record_files)
            todo_record_files = todo_record_files.subtract(completed_record_files).distinct()

        result = (
            # RDD(record_files)
            todo_record_files
            # RDD(0/1), 1 for success
            .map(lambda records_file: self.process_file(
                records_file,
                records_file.replace(origin_prefix, target_prefix, 1)))
            .cache())

        if result.isEmpty():
            logging.info("Nothing to be processed, everything is under control!")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_file(record_filepath, target_filepath):
        """Call convert python code to convert records"""
        try:
            file_utils.makedirs(os.path.dirname(target_filepath))
            convert_kinglong_to_apollo(record_filepath, target_filepath)
            logging.info('Successfully labeled {}'.format(record_filepath))
            return 1
        except BaseException as e:
            logging.error('Failed to process {}'.format(record_filepath))
        return 0


if __name__ == '__main__':
    ConvertKinglongToApollo().main()
