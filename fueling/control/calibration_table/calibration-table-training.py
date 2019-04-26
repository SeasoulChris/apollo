#!/usr/bin/env python
from collections import Counter
import glob
import operator
import os

import colored_glog as glog
import h5py
import numpy as np
import pyspark_utils.op as spark_op

from modules.common.configs.proto import vehicle_config_pb2
import common.proto_utils as proto_utils
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_train_utils as train_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as CalibrationTable


FILENAME_CALIBRATION_TABLE_CONF = \
    '/apollo/modules/data/fuel/fueling/control/conf/calibration_table_conf.pb.txt'
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           CalibrationTable.CalibrationTable())
FILENAME_VEHICLE_PARAM_CONF = '/apollo/modules/common/data/vehicle_param.pb.txt'
VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(FILENAME_VEHICLE_PARAM_CONF,
                                                       vehicle_config_pb2.VehicleConfig())

WANTED_VEHICLE = CALIBRATION_TABLE_CONF.vehicle_type

brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]
throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]
train_alpha = CALIBRATION_TABLE_CONF.train_alpha

brake_axis_cmd_min = -1 * CALIBRATION_TABLE_CONF.brake_max
brake_axis_cmd_max = -1 * VEHICLE_PARAM_CONF.vehicle_param.brake_deadzone

speed_min = CALIBRATION_TABLE_CONF.train_speed_min
speed_max = CALIBRATION_TABLE_CONF.train_speed_max
speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment

throttle_axis_cmd_min = VEHICLE_PARAM_CONF.vehicle_param.throttle_deadzone
throttle_axis_cmd_max = CALIBRATION_TABLE_CONF.throttle_max
cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment


def get_feature_hdf5_files(feature_dir, throttle_or_brake, train_or_test):
    return (
        # RDD(feature folder)
        feature_dir
        # RDD(throttle/brake train/test feature folder)
        .map(lambda feature_dir: os.path.join(feature_dir, throttle_or_brake, train_or_test))
        # RDD(all files in throttle train feature folder)
        .flatMap(lambda path: glob.glob(os.path.join(path, '*.hdf5')))
        # PairRDD('throttle or brake', hdf5 files)
        .keyBy(lambda _: throttle_or_brake)
        # PairRDD('throttle or brake', hdf5 files RDD)
        .groupByKey()
        # PairRDD('throttle or brake', list of hdf5 files)
        .mapValues(list))


def get_feature_hdf5_files_prod(bucket, feature_prefix, throttle_or_brake):
    return (
        # RDD(throttle feature folder)
        s3_utils.list_files(bucket, feature_prefix, '.hdf5')
        # PairRDD('throttle or brake', hdf5 files)
        .keyBy(lambda _: throttle_or_brake)
        # PairRDD('throttle or brake', hdf5 files RDD)
        .groupByKey()
        # PairRDD('throttle or brake', list of hdf5 files)
        .mapValues(list))


def get_data_from_hdf5(hdf5_rdd):
    return (
        # PairRDD('throttle or brake', list of hdf5 files)
        hdf5_rdd
        # PairRDD('throttle or brake', segments)
        .mapValues(train_utils.generate_segments)
        # PairRDD('throttle or brake', data)
        .mapValues(train_utils.generate_data))


class CalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_training')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_dir = os.path.join('/apollo/modules/data/fuel/testdata/control/generated',
                                  WANTED_VEHICLE, 'CalibrationTable')
        target_dir = os.path.join('/apollo/modules/data/fuel/testdata/control/generated',
                                  WANTED_VEHICLE, 'conf')
        # RDD(origin_dir)
        feature_dir = self.to_rdd([origin_dir])

        # RDD('throttle', list of hdf5 files)
        throttle_train_files = get_feature_hdf5_files(feature_dir, 'throttle', 'train')
        # RDD('throttle', list of hdf5 files)
        throttle_test_files = get_feature_hdf5_files(feature_dir, 'throttle', 'test')

        # RDD('brake', list of hdf5 files)
        brake_train_files = get_feature_hdf5_files(feature_dir, 'brake', 'train')
        # RDD('brake', list of hdf5 files)
        brake_test_files = get_feature_hdf5_files(feature_dir, 'brake', 'test')

        feature_dir_rdds = (throttle_train_files, throttle_test_files,
                            brake_train_files, brake_test_files)
        self.run(feature_dir_rdds, target_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_dir = bos_client.abs_path(
            os.path.join('modules/control/CalibrationTable/Features', WANTED_VEHICLE))
        target_dir = bos_client.abs_path(
            os.path.join('modules/control/CalibrationTable/Conf', WANTED_VEHICLE))
        throttle_train_prefix = os.path.join(origin_dir, 'throttle', 'train')
        # RDD('throttle', list of hdf5 files)
        throttle_train_files = get_feature_hdf5_files_prod(
            bucket, throttle_train_prefix, 'throttle')

        throttle_test_prefix = os.path.join(origin_dir, 'throttle', 'test')
        # RDD('throttle', list of hdf5 files)
        throttle_test_files = get_feature_hdf5_files_prod(bucket, throttle_test_prefix, 'throttle')

        brake_train_prefix = os.path.join(origin_dir, 'brake', 'train')
        # RDD('brake', list of hdf5 files)
        brake_train_files = get_feature_hdf5_files_prod(bucket, brake_train_prefix, 'brake')

        brake_test_prefix = os.path.join(origin_dir, 'brake', 'test')
        # RDD('brake', list of hdf5 files)
        brake_test_files = get_feature_hdf5_files_prod(bucket, brake_test_prefix, 'brake')

        feature_dir_rdds = (throttle_train_files, throttle_test_files,
                            brake_train_files, brake_test_files)
        self.run(feature_dir_rdds, target_dir)

    def run(self, feature_dir_rdds, target_dir):
        """ processing RDD """
        # RDD('throttle', list of train hdf5 files), RDD ('throttle', list of test hdf5 files),
        # RDD('brake', list of train hdf5 files), RDD ('brake', list of test hdf5 files)
        throttle_train_files, throttle_test_files, brake_train_files, brake_test_files = \
            feature_dir_rdds

        glog.info("throttle train file ONE: %s", throttle_train_files.first())
        glog.info("throttle test file ONE: %s", throttle_test_files.first())

        # PairRDD('throttle', train data)
        throttle_train_data = get_data_from_hdf5(throttle_train_files).cache()
        # PairRDD('throttle', test data)
        throttle_test_data = get_data_from_hdf5(throttle_test_files).cache()

        glog.info("throttle train data segment numbers: %d", throttle_train_data.count())
        glog.info("throttle test data segment numbers: %d", throttle_test_data.count())

        throttle_table_filename = 'throttle_calibration_table.pb.txt'
        throttle_model_rdd = (
            # PairRDD(dir, (x_train_data, y_train_data))
            throttle_train_data
            # PairRDD(dir, (x_train_data, y_train_data, x_test_data, y_test_data))
            .join(throttle_test_data)
            # PairRDD(dir, result_array)
            .mapValues(lambda elem:
                       train_utils.train_model(elem, throttle_train_layer, train_alpha))
            # RDD(a number)
            .map(lambda elem:
                 train_utils.write_table(elem, target_dir, speed_min, speed_max, speed_segment_num,
                                         throttle_axis_cmd_min, throttle_axis_cmd_max,
                                         cmd_segment_num, throttle_table_filename))
            .count())

        glog.info("brake train file ONE: %s", brake_train_files.first())
        glog.info("brake test file ONE: %s", brake_test_files.first())

        # PairRDD('brake', train data)
        brake_train_data = get_data_from_hdf5(brake_train_files).cache()
        # PairRDD('brake', test data)
        brake_test_data = get_data_from_hdf5(brake_test_files).cache()

        glog.info("brake train data numbers: %d", brake_train_data.count())
        glog.info("brake test data numbers: %d", brake_test_data.count())

        brake_table_filename = 'brake_calibration_table.pb.txt'
        brake_model_rdd = (
            # PairRDD(dir, (x_train_data, y_train_data))
            brake_train_data
            # PairRDD(dir, ((x_train_data, y_train_data), (x_test_data, y_test_data))
            .join(brake_test_data)
            # PairRDD(dir, result_array)
            .mapValues(lambda elem: train_utils.train_model(elem, brake_train_layer, train_alpha))
            # RDD(a number)
            .map(lambda (path, model):
                 train_utils.write_table(model, target_dir, speed_min, speed_max, speed_segment_num,
                                         brake_axis_cmd_min, brake_axis_cmd_max,
                                         cmd_segment_num, brake_table_filename))
            .count())


if __name__ == '__main__':
    CalibrationTableTraining().main()
