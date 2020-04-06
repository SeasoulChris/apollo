#!/usr/bin/env python
import operator
import os
import time

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.planning.datasets.label_generator import LabelGenerator
import fueling.common.logging as logging

SKIP_EXISTING_DST_FILE = False
SRC_DIR_PREFIX = 'modules/planning/learning_data'
DST_DIR_PREFIX = 'modules/planning/output_data'
# for local test
# SRC_DIR_PREFIX = 'apollo/data/learning_data'
# DST_DIR_PREFIX = 'apollo/data/output_data'


class OutputDataGenerator(BasePipeline):
    """output data for offline training"""

    # def __init__(self):
    #     self.src_dir_prefix = 'modules/planning/learning_data'

    def run(self):
        # RDD(bin_files)
        bin_files = (
            self.to_rdd(self.our_storage().list_files(SRC_DIR_PREFIX))
            .filter(spark_op.filter_path(['*.bin'])))
        # labeled_bin_files = (
        #     # RDD(label_files)
        #     self.to_rdd(self.our_storage().list_files(SRC_DIR_PREFIX))
        #     # RDD(bin_files)
        #     .map(lambda label_file: label_file.replace('.bin.future_status.npy', '.bin')))
        # RDD(todo_bin_files)
        todo_bin_files = bin_files
        logging.info(bin_files.collect())

        # if SKIP_EXISTING_DST_FILE:
        #     # RDD(todo_bin_files)
        #     todo_bin_files = todo_bin_files.subtract(labeled_bin_files).distinct()
        logging.info(todo_bin_files.count())
        self.run_internal(todo_bin_files)

    def run_internal(self, bin_files_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        logging.info(bin_files_rdd.count())
        # bin_files_rdd = bin_files_rdd.keyBy(lambda src_file: os.path.dirname(src_file))
        # logging.info(f'(file_dir, src_file): {bin_files_rdd.collect()}')
        # combine every 2 rdd files
        # Paired RDD(dir, (file_id,file_dir))
        bin_files_rdd_origin = bin_files_rdd.map(self.get_file_id).cache()
        # logging.info(bin_files_rdd_origin.collect())
        # get the id of the first file (when the file ID starts from non-zero)
        # combine file ids in a folder
        # find the minimum id in each folder
        min_fileID_rdd = (bin_files_rdd_origin.keys().reduceByKey(min))
        # reduceByKey(
        #     lambda dir_FileID: dir_FileID[1] >= 0))  # .reduceByKey(min)
        logging.info(min_fileID_rdd.collect())
        # logging.info(f'first file id in origin bin files: {grouped}')

        # Paired RDD(file_id-1, (dir, file_dir))
        bin_files_rdd_shifted = (
            bin_files_rdd
            .map(lambda elem: self.get_file_id(elem, True)))
        # remove the first file (with min_fileID) in shifted bin files rdd
        # for example:
        # origin RDD fileIDs are (37, 38, 39)
        # shift RDD fileIDs are (37-1, 38-1, 39-1) => (36, 37, 38)
        # and shifted file 37 is complete file for origin file 37;
        # shifted file 38 is complete file for origin file 38
        # and file 36 is an extra file.
        bin_files_rdd_shifted = (
            bin_files_rdd_shifted
            # PairedRDD (dir, (fileID, srcFile))
            .map(lambda dir_FileID_srcFile: (
                dir_FileID_srcFile[0][0], (dir_FileID_srcFile[0][1], dir_FileID_srcFile[1])))
            # PairedRDD (dir, ((fileID, srcFile), min_fileID))
            .join(min_fileID_rdd)
            # PairedRDD (dir, ((fileID, srcFile), min_fileID)) where fileID>min_fileID
            .filter(
                spark_op.filter_value(
                    lambda FileID_srcFile_minFileID: FileID_srcFile_minFileID[0][0]
                    >= FileID_srcFile_minFileID[1]))
            # PairedRDD (dir, (fileID, srcFile))
            .map(lambda dir_FileID_srcFile_minFileID:
                 ((dir_FileID_srcFile_minFileID[0],
                   dir_FileID_srcFile_minFileID[1][0][0]), dir_FileID_srcFile_minFileID[1][0][1]))
            .cache()
        )
        # Paired RDD( (file_dir, file_id), (file_dir,next_file_dir))
        bin_file_couple_rdd = (
            bin_files_rdd_origin
            .cogroup(bin_files_rdd_shifted)
            .mapValues(lambda bin1_bin2: (list(bin1_bin2[0]), list(bin1_bin2[1]))))
        logging.info(bin_file_couple_rdd.count())
        logging.info(bin_file_couple_rdd.keys().collect())
        logging.info(bin_file_couple_rdd.first())

        # Paired RDD(file_id, (origin_bin, next_bin))
        result = bin_file_couple_rdd.mapValues(self.process_file).cache()
        if result.isEmpty():
            logging.info("Nothing to be processed.")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def get_file_id(src_file, is_shift=False):
        # file name is learning_data.x.bin
        file_name = os.path.basename(src_file)
        # # using dir as part of key
        file_dir = os.path.dirname(src_file)
        logging.debug(file_name.split('.')[1])
        file_id = int(file_name.split('.')[1])
        if is_shift:
            file_id = file_id - 1
        return (file_dir, file_id), src_file

    @staticmethod
    def process_file(file_paths):
        """Call label_generator to generate labels."""
        logging.info(file_paths)
        src_file = file_paths[0][0]
        if len(file_paths[1]):
            secondary_file = file_paths[1][0]
        else:
            secondary_file = None
        logging.info(src_file)
        logging.info(secondary_file)
        label_gen = LabelGenerator()
        dst_file = src_file.replace(SRC_DIR_PREFIX, DST_DIR_PREFIX)
        dst_dir = os.path.dirname(dst_file)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        logging.info(dst_file)
        logging.info(dst_dir)
        try:
            label_gen.GetObserveAllFeatureSequences(src_file, dst_file, secondary_file)
            logging.info('Successfully load feature pb and save label files {}'.format(src_file))
            label_gen.Label()
            logging.info('Successfully labeled {}'.format(src_file))
            return 1
        except BaseException:
            logging.error('Failed to process {}'.format(src_file))
        return 0


if __name__ == '__main__':
    OutputDataGenerator().main()
