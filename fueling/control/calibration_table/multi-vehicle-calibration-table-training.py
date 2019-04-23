#!/usr/bin/env python
from collections import Counter
import glob
import operator
import os

import colored_glog as glog
import h5py
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from modules.common.configs.proto import vehicle_config_pb2
import common.proto_utils as proto_utils
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2

from fueling.common.base_pipeline import BasePipeline
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

throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]

brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]

train_alpha = CALIBRATION_TABLE_CONF.train_alpha


def get_vehicle_type(data_folder):
    sub_folders = os.listdir(data_folder)
    vehicle_types = []
    for one_folder in sub_folders:
        vehicle_types.append(one_folder.rsplit('/'))
    return vehicle_types


def get_todo_dirs(origin_vehicles):
    """ for run_test only, folder/vehicle/subfolder/*.record.* """
    return (origin_vehicles
            # PairRDD(vehicle, end_file_lists)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*')))
            .mapValues(os.path.dirname))


def get_vehicle_param(folder_dir):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(
        conf_file, vehicle_config_pb2.VehicleConfig())
    return VEHICLE_PARAM_CONF.vehicle_param


def get_feature_hdf5_files(feature_dir, throttle_or_brake, train_or_test):
    return (
        # RDD(feature folder)
        feature_dir
        # RDD(throttle/brake train/test feature folder)
        .mapValues(lambda feature_dir: os.path.join(feature_dir, throttle_or_brake, train_or_test))
        # RDD(all files in throttle train feature folder)
        .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*.hdf5')))
        # PairRDD('throttle or brake', hdf5 files)
        .keyBy(lambda (vehicle, features): (vehicle, throttle_or_brake))
        .mapValues(lambda (vehicle, path): path)
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


def gen_train_param(vehicle_param, throttle_or_brake):
    if throttle_or_brake == 'throttle':
        cmd_min = vehicle_param.throttle_deadzone
        cmd_max = CALIBRATION_TABLE_CONF.throttle_max
        layer = throttle_train_layer
    elif throttle_or_brake == 'brake':
        cmd_min = -1 * CALIBRATION_TABLE_CONF.brake_max
        cmd_max = -1 * vehicle_param.brake_deadzone
        layer = brake_train_layer
    speed_min = CALIBRATION_TABLE_CONF.train_speed_min
    speed_max = CALIBRATION_TABLE_CONF.train_speed_max
    speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment
    cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment
    train_alpha = CALIBRATION_TABLE_CONF.train_alpha
    return ((speed_min, speed_max, speed_segment_num),
            (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha)


class MultiCalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'multi_calibration_table_training')

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableFeature'
        conf_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableConf'

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.context().parallelize([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(get_vehicle_type)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type[0])
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type[0])), 3)

        # RDD(origin_dir)
        conf_vehicle_dir = spark_helper.cache_and_log(
            'conf_vehicle_dir',
            self.context().parallelize([conf_prefix])
            # RDD([vehicle_type])
            .flatMap(get_vehicle_type)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type[0])
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(conf_prefix, vehicle_type[0])), 3)

        """ get conf files """
        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # PairRDD(vehicle, dir_of_vehicle)
            conf_vehicle_dir
            # PairRDD(vehicle_type, vehicle_conf)
            .mapValues(get_vehicle_param), 3)

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_train_files = get_feature_hdf5_files(origin_vehicle_dir, 'throttle', 'train')

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_test_files = get_feature_hdf5_files(origin_vehicle_dir, 'throttle', 'test')

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_train_files = get_feature_hdf5_files(origin_vehicle_dir, 'brake', 'train')

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_test_files = get_feature_hdf5_files(origin_vehicle_dir, 'brake', 'test')
        glog.info(brake_test_files.collect())

        feature_dir = (throttle_train_files, throttle_test_files,
                       brake_train_files, brake_test_files)

        self.run(feature_dir, vehicle_param_conf, origin_prefix, target_prefix)

    def run(self, feature_dir, vehicle_param_conf, origin_prefix, target_prefix):
        throttle_train_files, throttle_test_files, brake_train_files, brake_test_files = feature_dir

        # PairRDD((vehicle, 'throttle'), train data)
        throttle_train_data = spark_helper.cache_and_log(
            'throttle_train_data',
            get_data_from_hdf5(throttle_train_files))
        # PairRDD((vehicle, 'throttle'), test data)
        throttle_test_data = spark_helper.cache_and_log(
            'throttle_test_data',
            get_data_from_hdf5(throttle_test_files))

        throttle_data = spark_helper.cache_and_log(
            'throttle_data',
            # PairRDD((vehicle, 'throttle'), (x_train_data, y_train_data))
            throttle_train_data
            # PairRDD((vehicle, 'throttle'), ((x_train_data, y_train_data), (x_test_data, y_test_data)))
            .join(throttle_test_data))

        throttle_train_param = spark_helper.cache_and_log(
            'throttle_train_param',
            # PairRDD(vehicle, train_param)
            vehicle_param_conf.mapValues(lambda conf: gen_train_param(conf, 'throttle'))
            # PairRDD((vehicle, 'throttle'), train_param)
            .map(lambda (vehicle, train_param): ((vehicle, 'throttle'), train_param)))

        throttle_model = spark_helper.cache_and_log(
            'throttle_model',
            # PairRDD((vehicle, 'throttle'), (data_set, train_param))
            throttle_data.join(throttle_train_param)
            # RDD(table_filename)
            .map(lambda elem: train_utils.train_write_model(elem, target_prefix)))

        """ brake """
        # PairRDD((vehicle, 'brake'), train data)
        brake_train_data = spark_helper.cache_and_log(
            'brake_train_data',
            get_data_from_hdf5(brake_train_files))
        # PairRDD((vehicle, 'brake'), test data)
        brake_test_data = spark_helper.cache_and_log(
            'brake_test_data',
            get_data_from_hdf5(brake_test_files))

        brake_data = spark_helper.cache_and_log(
            'brake_data',
            # PairRDD((vehicle, 'brake'), (x_train_data, y_train_data))
            brake_train_data
            # PairRDD((vehicle, 'brake'), ((x_train_data, y_train_data), (x_test_data, y_test_data)))
            .join(brake_test_data))

        brake_train_param = spark_helper.cache_and_log(
            'brake_train_param',
            # PairRDD(vehicle, train_param)
            vehicle_param_conf.mapValues(lambda conf: gen_train_param(conf, 'brake'))
            # PairRDD((vehicle, 'brake'), train_param)
            .map(lambda (vehicle, train_param): ((vehicle, 'brake'), train_param)))

        brake_model = spark_helper.cache_and_log(
            'brake_model',
            # PairRDD((vehicle, 'brake'), (data_set, train_param))
            brake_data.join(brake_train_param)
            # RDD(table_filename)
            .map(lambda elem: train_utils.train_write_model(elem, target_prefix)))


if __name__ == '__main__':
    MultiCalibrationTableTraining().main()
