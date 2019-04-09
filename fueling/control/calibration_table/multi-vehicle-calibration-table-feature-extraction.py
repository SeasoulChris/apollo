#!/usr/bin/env python
""" extract features for multiple vehicles """
from collections import Counter
import operator
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from modules.common.configs.proto.vehicle_config_pb2 import VehicleParam
import common.proto_utils as proto_utils

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import gen_pre_segment
from fueling.control.features.feature_extraction_utils import pair_cs_pose
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
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

def get_vehicle_param(folder_dir, vehicle_type):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_type, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(conf_file, VehicleParam())
    return VEHICLE_PARAM_CONF

def gen_target_prefix(root_dir, vehicle_type):
    return  vehicle_type, os.path.join(root_dir, 'modules/data/fuel/testdata/control/generated',
                         vehicle_type, 'CalibrationTable')

def gen_throttle_train_target_prefix(vehicle_type):
    return os.path.join('modules/data/fuel/testdata/control/generated',
                        vehicle_type, 'CalibrationTable', 'throttle', 'train')

def re_org_dir(elem):
    (vehicle_type, (origin_dir, target_dir)), vehicle_conf = elem
    return ((vehicle_type, vehicle_conf), (origin_dir, target_dir)) 

    
# def list_to_do_tasts(vehicle_dir, origin_dir, target_dir):
#     throttle_train_target_dir = os.path.join(target_dir, 'throttle', 'train')
#     return dir_utils.list_end_files(vehicle_dir)

#     list_func = (lambda path: self.get_spark_context().parallelize(
#             dir_utils.list_end_files(origin_dir)))
#     # RDD(record_dir)
#         todo_tasks = (
#             dir_utils.get_todo_tasks(
#                 origin_prefix, throttle_train_target_prefix, list_func, '', '/' + MARKER))

#         glog.info('todo_folders: {}'.format(todo_tasks.collect()))

#         dir_to_records = (
#             # PairRDD(record_dir, record_dir)
#             todo_tasks
#             # PairRDD(record_dir, all_files)
#             .flatMap(dir_utils.list_end_files)
#             # PairRDD(record_dir, record_files)
#             .filter(record_utils.is_record_file)
#             # PairRDD(record_dir, record_files)
#             .keyBy(os.path.dirname))

#         glog.info('todo_files: {}'.format(dir_to_records.collect()))

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

        target_prefix_rdd = (
            # PairRDD(vehicle_type, path_to_vehicle_type)
            origin_dir_rdd
            # PairRDD(vehicle_type, target_prefix)
            .map(lambda vehicleType_folder: gen_target_prefix(root_dir, vehicleType_folder[0]))
            .cache())
        
        # PairRDD(vehicle_type, (origin_prefix, target_prefix))
        origin_target_dir_rdd = origin_dir_rdd.join(target_prefix_rdd)

        print(origin_target_dir_rdd.collect())

        with_conf_file = (
            # PairRDD(vehicle_type, (origin_prefix, target_prefix))
            origin_target_dir_rdd
             # PairRDD((vehicle_type, (origin_prefix, target_prefix)), vehicle_conf)
            .map(lambda elem:(elem, get_vehicle_param(elem[0], elem[1][0])))
            .map(re_org_dir)
            )

        print(with_conf_file.first())




if __name__ == '__main__':
    CalibrationTableFeatureExtraction().run_test()