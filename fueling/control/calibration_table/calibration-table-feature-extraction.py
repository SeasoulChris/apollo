#!/usr/bin/env python
"""This is a module to extraction features from records with folder path as part of the key"""

from collections import Counter
import operator
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from modules.common.configs.proto.vehicle_config_pb2 import VehicleParam

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import gen_pre_segment
from fueling.control.features.feature_extraction_utils import pair_cs_pose
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 10
MARKER = 'CompleteCalibrationTable'

def get_single_vehicle_type(data_folder):
    sub_folders = os.listdir(data_folder)
    for one_folder in sub_folders:
        vehicle_type = one_folder.rsplit('/')
    return vehicle_type[0]

def get_vehicle_param(folder_dir, vehicle_type):
    # CONF_FOLDER = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_type, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(conf_file, VehicleParam())
    return VEHICLE_PARAM_CONF

class CalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        WANTED_VEHICLE = get_single_vehicle_type(origin_prefix)
        VEHICLE_PARAM_CONF = get_vehicle_param(origin_prefix, WANTED_VEHICLE)
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'CalibrationTable')
        throttle_train_target_prefix = os.path.join(target_prefix, 'throttle', 'train')
        # RDD(record_dirs)
        todo_tasks = self.to_rdd([origin_prefix])
        # PairRDD(record_dirs, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
            dir_utils.get_todo_records(todo_tasks))
        self.run(todo_records, origin_prefix, target_prefix, 
            throttle_train_target_prefix, VEHICLE_PARAM_CONF)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join('modules/control/CalibrationTable/', WANTED_VEHICLE)
        throttle_train_target_prefix = os.path.join(target_prefix, 'throttle', 'train')
        # RDD(record_dirs)
        todo_tasks = dir_utils.get_todo_tasks(origin_prefix, target_prefix, 'COMPLETE', MARKER)
        # PairRDD(record_dir, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
            dir_utils.get_todo_records(todo_tasks))
        self.run(todo_records, origin_prefix, target_prefix, throttle_train_target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, 
            throttle_train_target_prefix, VEHICLE_PARAM_CONF):
        """ processing RDD """
        # PairRDD((dir, timestamp_per_min), msg)
        dir_to_msgs = (dir_to_records_rdd
                      # PairRDD(dir, msg)
                      .flatMapValues(record_utils.read_record(channels))
                      # PairRDD((dir, timestamp_per_min), msg)
                      .map(gen_pre_segment)
                      .cache())

        # RDD(dir, timestamp_per_min)
        valid_segments = (feature_extraction_rdd_utils.
                            chassis_localization_segment_rdd(dir_to_msgs, MIN_MSG_PER_SEGMENT))

        # PairRDD((dir_segment, segment_id), msg) 
        valid_msgs = feature_extraction_rdd_utils.valid_msg_rdd(dir_to_msgs, valid_segments)
 
        # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
        parsed_msgs = feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msgs)

        calibration_table_rdd = (
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            parsed_msgs
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(pair_cs_pose)
            # PairRDD((dir_segment, segment_id), features)
            .mapValues(
                lambda msgs:calibration_table_utils.feature_generate(msgs, VEHICLE_PARAM_CONF))
            # PairRDD ((dir_segment, segment_id), filtered_features)
            .mapValues(calibration_table_utils.feature_filter)
            # PairRDD ((dir_segment, segment_id), cutted_features)
            .mapValues(
                lambda feature: calibration_table_utils.feature_cut(feature, VEHICLE_PARAM_CONF))
            # PairRDD ((dir_segment, segment_id), (grid_dict,cutted_features))
            .mapValues(lambda feature:
                       calibration_table_utils.feature_distribute(feature, VEHICLE_PARAM_CONF))
            # PairRDD ((dir_segment, segment_id), one_matrix)
            .mapValues(
                lambda feature: calibration_table_utils.feature_store(feature, VEHICLE_PARAM_CONF))
            # RDD(feature_numbers)
            .map(lambda elem:
                 calibration_table_utils.write_h5_train_test(elem, origin_prefix, target_prefix)))

        glog.info('Finished %d calibration_table_rdd!' % calibration_table_rdd.count())

        # RDD (dir_segment)
        (feature_extraction_rdd_utils.mark_complete(valid_msgs, origin_prefix,
                                                    throttle_train_target_prefix, MARKER)
         .count())


if __name__ == '__main__':
    CalibrationTableFeatureExtraction().main()
