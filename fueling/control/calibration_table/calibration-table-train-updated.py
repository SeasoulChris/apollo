#!/usr/bin/env python
from collections import Counter
import glob
import h5py
import operator
import os

import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from modules.common.configs.proto import vehicle_config_pb2
import common.proto_utils as proto_utils
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_train_utils as train_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2
import modules.control.proto.control_conf_pb2 as ControlConf
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as CalibrationTable


FILENAME_CALIBRATION_TABLE_CONF = \
    '/apollo/modules/data/fuel/fueling/control/conf/calibration_table_conf.pb.txt'
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           CalibrationTable.CalibrationTable())
FILENAME_CONTROL_CONF = '/apollo/modules/calibration/data/transit/control_conf.pb.txt'
CONTROL_CONF = proto_utils.get_pb_from_text_file(FILENAME_CONTROL_CONF, ControlConf.ControlConf())
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

# TODO: list func

def get_feature_hdf5_files(feature_dir, root_dir, throttle_or_brake, train_or_test, list_func):
    return (
        #RDD(feature folder)
        feature_dir
        #RDD(throttle train feature folder)
        .map(lambda feature_dir: os.path.join(root_dir, feature_dir, throttle_or_brake, train_or_test))
        #RDD(all files in throttle train feature folder)
        .flatMap(list_func)
        #RDD(hdf5 files)
        .filter(lambda path: path.endswith('.hdf5'))
        #PairRDD('throttle or brake', hdf5 files)
        .map(lambda hdf5_files: (throttle_or_brake, hdf5_files))
        #PairRDD('throttle or brake', hdf5 files RDD)
        .groupByKey()
        #PairRDD('throttle or brake', list of hdf5 files)
        .mapValues(list))

def get_data_from_hdf5(hdf5_rdd):
    return (
        #PairRDD('throttle or brake', list of hdf5 files)
        hdf5_rdd
        #PairRDD('throttle or brake', segments)
        .mapValues(train_utils.generate_segments)
        #PairRDD('throttle or brake', data)
        .mapValues(train_utils.generate_data))
    
class CalibrationTableTraining(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_training')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_prefix = os.path.join('modules/data/fuel/testdata/control/generated/', 
                            WANTED_VEHICLE, 'CalibrationTable')
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated/', 
                            WANTED_VEHICLE, 'conf')
        root_dir = '/apollo'
        target_dir = os.path.join(root_dir, target_prefix)
        # RDD(origin_prefix)
        calibration_table_feature_dir = self.get_spark_context().parallelize([origin_prefix])
        list_func = dir_utils.list_end_files
        self.run(calibration_table_feature_dir, root_dir, target_dir, list_func)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = os.path.join('modules/control/CalibrationTable/Features', WANTED_VEHICLE)
        target_prefix = os.path.join('modules/control/CalibrationTable/Conf', WANTED_VEHICLE)
        root_dir = s3_utils.S3_MOUNT_PATH

        target_dir = os.path.join(root_dir, target_prefix)

        list_func = (lambda path: s3_utils.list_files(bucket, origin_prefix))

        # RDD(origin_prefix)
        calibration_table_feature_dir = self.get_spark_context().parallelize([origin_prefix])

        self.run(calibration_table_feature_dir, root_dir, target_dir, list_func)


    def run(self, feature_dir, root_dir, target_dir, list_func):
        """ processing RDD """
        throttle_train_files = (
            # RDD ('throttle', list of hdf5 files)
            get_feature_hdf5_files(feature_dir, root_dir, 'throttle', 'train', list_func))
        throttle_test_files = (
            # RDD ('throttle', list of hdf5 files)
            get_feature_hdf5_files(feature_dir, root_dir, 'throttle', 'test', list_func))

        glog.info("throttle test file ONE: %s", throttle_test_files.first())

        throttle_train_data = get_data_from_hdf5(throttle_train_files).cache()
        throttle_test_data = get_data_from_hdf5(throttle_test_files).cache()
            
        glog.info("throttle train data segment numbers: %d", throttle_train_data.count())
        glog.info("throttle test data segment numbers: %d", throttle_test_data.count())


        throttle_table_filename = 'throttle_calibration_table.pb.txt'
        throttle_model_rdd = (
            # PairRDD (dir, (x_train_data, y_train_data))
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

        brake_train_files = (
            # RDD ('brake', list of hdf5 files)
            get_feature_hdf5_files(feature_dir, root_dir, 'brake', 'train', list_func))
        brake_test_files = (
            # RDD ('brake', list of hdf5 files)
            get_feature_hdf5_files(feature_dir, root_dir, 'brake', 'test', list_func))

        glog.info("brake train file ONE: %s", brake_train_files.first())
        glog.info("brake test file ONE: %s", brake_test_files.first())

        brake_train_data = get_data_from_hdf5(brake_train_files).cache()
        brake_test_data = get_data_from_hdf5(brake_test_files).cache()
            
        glog.info("brake train data numbers: %d", brake_train_data.count())
        glog.info("brake test data numbers: %d", brake_test_data.count())

        brake_table_filename = 'brake_calibration_table.pb.txt'
        brake_model_rdd = (
            # PairRDD (dir, (x_train_data, y_train_data))
            brake_train_data
            # PairRDD(dir, (x_train_data, y_train_data, x_test_data, y_test_data))
            .join(brake_test_data)
            # PairRDD(dir, result_array)
            .mapValues(lambda elem:
                       train_utils.train_model(elem, brake_train_layer, train_alpha))
            # RDD(a number)
            .map(lambda elem:
                 train_utils.write_table(elem, target_dir, speed_min, speed_max, speed_segment_num,
                                         brake_axis_cmd_min, brake_axis_cmd_max,
                                         cmd_segment_num, brake_table_filename))
            .count())


        
if __name__ == '__main__':
    CalibrationTableTraining().main()
