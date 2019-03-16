#!/usr/bin/env python
from collections import Counter
import glob
import h5py
import operator
import os

import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import common.proto_utils as proto_utils
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_train_utils as train_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2
import modules.control.proto.control_conf_pb2 as ControlConf
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as calibrationTable


WANTED_VEHICLE = 'Transit'

FILENAME_CALIBRATION_TABLE_CONF = \
    '/apollo/modules/data/fuel/fueling/control/conf/calibration_table_conf.pb.txt'
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           calibrationTable.calibrationTable())

FILENAME_CONTROL_CONF = \
    '/mnt/bos/code/apollo-internal/modules_data/calibration/data/transit/control_conf.pb.txt'
CONTROL_CONF = proto_utils.get_pb_from_text_file(
    FILENAME_CONTROL_CONF, ControlConf.ControlConf())


brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]
throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]
train_alpha = CALIBRATION_TABLE_CONF.train_alpha

brake_axis_cmd_min = -1*CALIBRATION_TABLE_CONF.brake_max
brake_axis_cmd_max = -1*CONTROL_CONF.lon_controller_conf.brake_deadzone

speed_min = CALIBRATION_TABLE_CONF.train_speed_min
speed_max = CALIBRATION_TABLE_CONF.train_speed_max
speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment

throttle_axis_cmd_min = CONTROL_CONF.lon_controller_conf.throttle_deadzone
throttle_axis_cmd_max = CALIBRATION_TABLE_CONF.throttle_max
cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment


class CalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_training')

    def run_test(self):
        """Run test."""
        records = ['modules/data/fuel/testdata/control/']

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'
        dir_to_records = self.get_spark_context().parallelize(records).keyBy(os.path.dirname)
        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'

        # choose folder for wanted vehicle
        origin_prefix = os.path.join('modules/control/feature_extraction_hf5/2019/', WANTED_VEHICLE)
        # TODO: The target_prefix is not used finally.
        target_prefix = 'modules/control/calibration_table/'
        root_dir = s3_utils.S3_MOUNT_PATH
        # TODO: I have to change it like this to fit the followed pipeline. But it's not a good way
        # to use S3 storage.
        dir_to_h5s = (
            self.get_spark_context().parallelize(['modules/control/feature_extraction_hf5/2019/'])
            .keyBy(os.path.dirname))
        self.run(dir_to_h5s, origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """

        # -> (dir, record), in absolute path
        dir_to_records = dir_to_records_rdd.map(lambda x: (os.path.join(root_dir, x[0]),
                                                           os.path.join(root_dir, x[1]))).cache()
        # TODO: Go through the whole logic carefully.
        # 1. Choose better variable names. Many of them mismatched what they are.
        # 2. Remove redundant items, for example, the dir_to_records[1] is never used.
        throttle_train_file_rdd = (
            # (dir, dir)
            dir_to_records
            # -> (dir, hdf5_files)
            .map(lambda elem:
                 train_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'train'))
            # -> (dir, segments)
            .mapValues(train_utils.generate_segments)
            # -> (dir, x_train_data, y_train_data)
            .mapValues(train_utils.generate_data))

        throttle_test_file_rdd = (
            # (dir, dir)
            dir_to_records
            # -> (dir, hdf5_files)
            .map(lambda elem:
                 train_utils.choose_data_file(elem, WANTED_VEHICLE, 'throttle', 'test'))
            # -> (dir, segments)
            .mapValues(train_utils.generate_segments)
            # -> (dir, x_test_data, y_test_data)
            .mapValues(train_utils.generate_data))

        # TODO: Use subfolders instead of concat string. It's easier if you want to parse it back.
        throttle_table_filename = WANTED_VEHICLE + '_throttle_calibration_table.pb.txt'

        throttle_model_rdd = (
            # TODO: Refine the comments to describe accrurately. It's
            # (dir, (x_train_data, y_train_data)) here, not a 3-elements tuple.
            # (dir, x_train_data, y_train_data)
            throttle_train_file_rdd
            # -> (dir, x_train_data, y_train_data, x_test_data, y_test_data)
            .join(throttle_test_file_rdd)
            # -> (dir, result_array)
            .mapValues(lambda elem:
                       train_utils.train_model(elem, throttle_train_layer, train_alpha))
            # -> (a number)
            .map(lambda elem:
                 train_utils.write_table(elem, speed_min, speed_max, speed_segment_num,
                                         throttle_axis_cmd_min, throttle_axis_cmd_max,
                                         cmd_segment_num, throttle_table_filename))
            .count())

        brake_train_file_rdd = (
            # (dir, dir)
            dir_to_records
            # -> (dir, hdf5_files)
            .map(lambda elem: train_utils.choose_data_file(elem, WANTED_VEHICLE, 'brake', 'train'))
            # -> (dir, segments)
            .mapValues(train_utils.generate_segments)
            # -> (dir, x_train_data, y_train_data)
            .mapValues(train_utils.generate_data))

        brake_test_file_rdd = (
            # (dir, dir)
            dir_to_records
            # -> (dir, hdf5_files)
            .map(lambda elem: train_utils.choose_data_file(elem, WANTED_VEHICLE, 'brake', 'test'))
            # -> (dir, segments)
            .mapValues(train_utils.generate_segments)
            # -> (dir, x_train_data, y_train_data)
            .mapValues(train_utils.generate_data))

        brake_table_filename = WANTED_VEHICLE + '_brake_calibration_table.pb.txt'

        brake_model_rdd = (
            # (dir, x_train_data, y_train_data)
            brake_train_file_rdd
            # -> (dir, x_train_data, y_train_data, x_test_data, y_test_data)
            .join(brake_test_file_rdd)
            # -> (dir, result_array)
            .mapValues(lambda elem: train_utils.train_model(elem, brake_train_layer, train_alpha))
            # -> (a number)
            .map(lambda elem:
                 train_utils.write_table(elem, speed_min, speed_max, speed_segment_num,
                                         brake_axis_cmd_min, brake_axis_cmd_max, cmd_segment_num,
                                         brake_table_filename))
            .count())


if __name__ == '__main__':
    CalibrationTableTraining().run_prod()
