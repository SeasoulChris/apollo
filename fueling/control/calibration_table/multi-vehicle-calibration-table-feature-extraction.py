#!/usr/bin/env python
""" extract features for multiple vehicles """
from collections import Counter
import operator
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline
# from fueling.control.features.feature_extraction_utils import gen_pre_segment
from fueling.control.features.feature_extraction_utils import pair_cs_pose
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 1
MARKER = 'CompleteCalibrationTable'

def get_single_vehicle_type(data_folder):
    sub_folders = os.listdir(data_folder)
    vehicle_types = []
    for one_folder in sub_folders:
        vehicle_types.append(one_folder.rsplit('/'))
    return vehicle_types

def get_vehicle_param(folder_dir):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(conf_file, vehicle_config_pb2.VehicleConfig())
    return VEHICLE_PARAM_CONF.vehicle_param

def gen_target_prefix(root_dir, vehicle_type):
    return  vehicle_type, os.path.join(root_dir, 'modules/data/fuel/testdata/control/generated',
                         vehicle_type, 'CalibrationTable')

def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    (vehicle, task_dir), msg = dir_to_msg
    dt = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = dt.strftime('%Y%m%d-%H%M')
    return ((vehicle, task_dir, segment_id), msg)

class CalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        origin_prefix = 'modules/data/fuel/testdata/control/sourceData/OUT'
        origin_dir = os.path.join(root_dir,origin_prefix)

        # RDD(origin_dir)
        origin_dir_rdd = (self.get_spark_context().parallelize([origin_dir])
                    # RDD([vehicle_type])
                   .flatMap(get_single_vehicle_type)
                    # PairRDD(vehicle_type, [vehicle_type])
                   .keyBy(lambda vehicle_type: vehicle_type[0])
                   # PairRDD(vehicle_type, path_to_vehicle_type)
                   .mapValues(lambda vehicle_type: os.path.join(origin_dir,vehicle_type[0]))
                   .cache())
        # print(origin_dir_rdd.collect())


        target_dir_rdd = (
            # PairRDD(vehicle_type, path_to_vehicle_type)
            origin_dir_rdd
            # PairRDD(vehicle_type, target_prefix)
            .map(lambda vehicleType_folder: gen_target_prefix(root_dir, vehicleType_folder[0]))
            .cache())
        # PairRDD(vehicle_type, (target_prefix, origin_prefix))
        target_origin_dirs= target_dir_rdd.join(origin_dir_rdd)
        # print(target_origin_dirs.collect())

        # skipped the processed folders
        # PairRDD(vehicle, dir)
        origin_dirs = (
            origin_dir_rdd
            .flatMapValues(dir_utils.list_end_files)
            .mapValues(os.path.dirname)
            .cache())
        
        processed_dirs = (
            # PairRDD(vehicle_type, (target_prefix, origin_prefix))
            target_origin_dirs
            # PairRDD((vehicle_type, origin_prefix), target_prefix)
            .map(lambda vehicle_target_origin:
                 (vehicle_target_origin,vehicle_target_origin[1][0]))
            # PairRDD((vehicle_type, origin_prefix), files_with_target_prefix)
            .flatMapValues(dir_utils.list_end_files)
            # PairRDD((vehicle_type, origin_prefix), files_with_MARKER_with_target_prefix)
            .filter(lambda key_path: key_path[1].endswith(MARKER))
            # PairRDD(vehicle_type, files_with_MARKER_with_origin_prefix)
            .map(lambda key_path:
                (key_path[0][0], key_path[1].replace(key_path[0][1][0], key_path[0][1][1])))
            # PairRDD(vehicle_type, dir_include_MARKER_with_origin_prefix)
            .mapValues(os.path.dirname))

        # PairRDD(vehicle_type, dir_of_todos_with_origin_prefix)
        todo_dirs= origin_dirs.subtract(processed_dirs)
        # print(origin_dirs.collect())
        # print(processed_dirs.collect())
        # print(todo_dirs.collect())

        vehicle_param_conf = (
            # PairRDD(vehicle, dir_of_vehicle)
            origin_dir_rdd
             # PairRDD(vehicle_type, vehicle_conf)
            .mapValues(get_vehicle_param))
        # print(vehicle_param_conf.collect())

        todo_dir_with_conf = (todo_dirs.join(vehicle_param_conf)
            .map(lambda vehicle_path_conf: 
                 ((vehicle_path_conf[0], vehicle_path_conf[1][1]), vehicle_path_conf[1][0])))
        # print(todo_dir_with_conf.collect())

        self.run(todo_dirs, vehicle_param_conf, target_dir_rdd)

    def run(self, todo_dirs, vehicle_param_conf, target_dir_rdd):

        # records
        records = (todo_dirs
        # PairRDD(vehicle, files)
        .flatMapValues(dir_utils.list_end_files)
        # PairRDD(vehicle, records)
        .filter(lambda elem: record_utils.is_record_file(elem[1]))
        # PairRDD(vehicle, (dir, records))
        .mapValues(lambda records: (os.path.dirname(records), records))
        # PairRDD((vehicle, dir), records)
        .map(lambda vehicle_dir_file: 
            ((vehicle_dir_file[0], vehicle_dir_file[1][0]), vehicle_dir_file[1][1]))
        )
        print(records.first())

        msgs = (records
                # PairRDD(vehicle, msg)
                .flatMapValues(record_utils.read_record(channels))
                # PairRDD((vehicle, timestamp_per_min), msg)
                .map(gen_pre_segment)
                .cache())
        # elem = msgs.first() 
        # print(elem)

        # RDD(dir, timestamp_per_min)
        valid_segments = (feature_extraction_rdd_utils.
                            chassis_localization_segment_rdd(msgs, MIN_MSG_PER_SEGMENT))
        # PairRDD((dir_segment, segment_id), msg) 
        valid_msgs = feature_extraction_rdd_utils.valid_msg_rdd(msgs, valid_segments)

        print(valid_msgs.first())

        # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
        parsed_msgs = (feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msgs)
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(pair_cs_pose))


        elem = parsed_msgs.first() 
        print(elem[1]) # key

        # join conf file
        msgs_with_conf = (
            parsed_msgs.map(lambda key_value: (key_value[0][0], key_value[0], key_value[1])))
        print(msgs_with_conf.first())
        # calibration_table_rdd = (
        #     parsed_msgs
        #     .mapValues(
        #         lambda msgs:calibration_table_utils.feature_generate(msgs, VEHICLE_PARAM_CONF))
        #     # PairRDD ((dir_segment, segment_id), filtered_features)
        #     .mapValues(calibration_table_utils.feature_filter)
        #     # PairRDD ((dir_segment, segment_id), cutted_features)
        #     .mapValues(
        #         lambda feature: calibration_table_utils.feature_cut(feature, VEHICLE_PARAM_CONF))
        #     # PairRDD ((dir_segment, segment_id), (grid_dict,cutted_features))
        #     .mapValues(
        #         lambda feature: calibration_table_utils.feature_distribute(feature, VEHICLE_PARAM_CONF))
        #     # PairRDD ((dir_segment, segment_id), one_matrix)
        #     .mapValues(
        #         lambda feature: calibration_table_utils.feature_store(feature, VEHICLE_PARAM_CONF))
        # )


        
if __name__ == '__main__':
    CalibrationTableFeatureExtraction().run_test()