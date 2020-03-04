#!/usr/bin/env python
import operator
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
from fueling.common.record.kinglong.cybertron.python.convert import convert_kinglong_to_apollo


SKIP_EXISTING_DST_FILE = True


class ConvertKinglongToApollo(BasePipeline):
    """Convert Kinglong record to Apollo pipeline."""

    def run(self):
        origin_prefix = 'kinglong/data_20200226/'
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
            .map(lambda output_file: os.path.join(os.path.dirname(output_file), "../../..",
                                                  os.path.basename(output_file)))
            .map(os.path.abspath)
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
            .map(lambda records_filepath: self.process_file(
                records_filepath,
                self.get_target_filepath(records_filepath, origin_prefix, target_prefix)))
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

    def get_target_filepath(self, record_filepath, origin_prefix, target_prefix):
        """Return the target_filepath"""
        dirname = os.path.dirname(record_filepath) + "/"
        basename = os.path.basename(record_filepath)
        map_name = basename.split("-")[-1].split(".")[0]
        date = basename.split("_")[1][:8]
        vehicle_name = basename.split("_")[0]
        target_filepath = os.path.join(dirname.replace(origin_prefix, target_prefix, 1),
                                       map_name, date, vehicle_name, basename)
        return target_filepath


if __name__ == '__main__':
    ConvertKinglongToApollo().main()
