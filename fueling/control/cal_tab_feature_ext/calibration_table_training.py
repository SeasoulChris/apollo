import glob
import h5py
from collections import Counter
import operator
import os


import numpy as np
import pyspark_utils.op as spark_op

from neural_network_tf import NeuralNetworkTF
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.colored_glog as glog
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils


WANTED_VEHICLE = 'Transit'

MIN_MSG_PER_SEGMENT = 1


class CalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_training')

    def run_test(self):
        """Run test."""
        records = [
            'modules/data/fuel/testdata/control/calibration_table/Transit_throttle_test_20190130-1637.hdf5']

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'
        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)

        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """

        # -> (dir, record), in absolute path
        dir_to_records = dir_to_records_rdd.map(lambda x: (os.path.join(root_dir, x[0]),
                                                           os.path.join(root_dir, x[1]))).cache()

        train_file_rdd = (dir_to_records
                          # training data (hdf5 file) vehicle
                          .map(lambda elem:
                               calibration_table_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'train'))
                          # generate training data segment
                          .mapValues(calibration_table_utils.generate_segments)
                          #   generate training data: x_train_data, y_train_data
                          .mapValues(calibration_table_utils.generate_data)).cache()

        test_file_rdd = (dir_to_records
                         # training data (hdf5 file) vehicle
                         .map(lambda elem:
                              calibration_table_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'test'))
                         # generate training data segment
                         .mapValues(calibration_table_utils.generate_segments)
                         #   generate training data: x_train_data, y_train_data
                         .mapValues(calibration_table_utils.generate_data)).cache()

        train_model_rdd = (train_file_rdd
                           .join(test_file_rdd)
                           .mapValues())

        print(train_model_rdd.first())
    # train model(train data, test data)

    # write table


if __name__ == '__main__':
    CalibrationTableTraining().run_test()
