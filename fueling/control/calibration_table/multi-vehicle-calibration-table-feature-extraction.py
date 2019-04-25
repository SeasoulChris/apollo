#!/usr/bin/env python
""" extract features for multiple vehicles """
from collections import Counter
import glob
import operator
import os

import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
MIN_MSG_PER_SEGMENT = 1
MARKER = 'CompleteCalibrationTable'
TASK = 'CalibrationTalbe'


def list_hdf5_prod(path):
    bucket = 'apollo-platform'
    return s3_utils.list_files(bucket, path, '.hdf5').collect()


def get_vehicle_type(data_folder):
    sub_folders = os.listdir(data_folder)
    vehicle_types = []
    for one_folder in sub_folders:
        vehicle_types.append(one_folder.rsplit('/'))
    return vehicle_types


def get_vehicle_type_prod(prefix):
    bucket = 'apollo-platform'
    vehicle = []
    # list from RDD(conf_files).collect()
    vehicle_dirs = s3_utils.list_files(bucket, prefix, '.pb.txt').collect()
    for vehicle_dir in vehicle_dirs:
        path = vehicle_dir.split('/')
        vehicle.append(path[-2])
    return vehicle


def list_end_files_prod(prefix):
    bucket = 'apollo-platform'
    # list from RDD(files).collect()
    return s3_utils.list_files(bucket, prefix).collect()


def get_relative_path(abs_path):
    return abs_path.replace(s3_utils.BOS_MOUNT_PATH + '/', '', 1)


def get_vehicle_param_prod(prefix):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    bucket = 'apollo-platform'
    return(
        s3_utils.list_files(bucket, prefix, vehicle_para_conf_filename)
        # PairRDD(vehicle, conf_file_path)
        .keyBy(lambda path: path.split('/')[-2])
        # PairRDD(vehicle, conf)
        .mapValues(lambda conf_file: proto_utils.get_pb_from_text_file(
            conf_file, vehicle_config_pb2.VehicleConfig()))
        # PairRDD(vehicle, vehicle_param)
        .mapValues(lambda vehicle_conf: vehicle_conf.vehicle_param))


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


def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    (vehicle, task_dir), msg = dir_to_msg
    dt = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = dt.strftime('%Y%m%d-%H%M')
    return ((vehicle, task_dir, segment_id), msg)


def valid_segment(records):
    msg_segment = spark_helper.cache_and_log(
        'Msgs',
        records
        # PairRDD(vehicle, msg)
        .flatMapValues(record_utils.read_record(channels))
        # PairRDD((vehicle, task_dir, timestamp_per_min), msg)
        .map(gen_pre_segment))
    valid_msg_segment = spark_helper.cache_and_log(
        'valid_segments',
        # PairRDD((vehicle, task_dir, segment_id), msg)
        feature_extraction_rdd_utils.chassis_localization_segment_rdd(
            msg_segment, MIN_MSG_PER_SEGMENT))
    # PairRDD((vehicle, dir_segment, segment_id), msg)
    return spark_op.filter_keys(msg_segment, valid_msg_segment)


def write_h5(elem, origin_prefix, target_prefix):
    (vehicle, segment_dir, segment_id), feature_data = elem
    origin_prefix = os.path.join(origin_prefix, vehicle)
    target_prefix = os.path.join(target_prefix, 'CalibrationTableFeature', vehicle)
    data_set = ((segment_dir, segment_id), feature_data)
    # return feature_data
    return calibration_table_utils.write_h5_train_test(data_set, origin_prefix, target_prefix)


def mark_complete(valid_segment, origin_prefix, target_prefix, MARKER):
   # PairRDD((vehicle, segment_dir, segment_id), msg)
    result_rdd = (
        valid_segment
        # PairRDD(dir_segment, vehicle)
        .map(lambda ((vehicle, segment_dir, segment_id), msgs):
             (segment_dir, vehicle))
        # PariRDD(dir_segment) unique dir
        .distinct()
        # RDD(dir_segment with target_prefix)
        .map(lambda (path, vehicle):
             path.replace(os.path.join(origin_prefix, vehicle),
                          os.path.join(target_prefix, 'CalibrationTableFeature',
                                       vehicle, 'throttle', 'train'), 1))
        # RDD(MARKER files)
        .map(lambda path: os.path.join(path, MARKER)))
    glog.info(result_rdd.collect())
    # RDD(dir_MARKER)
    result_rdd.foreach(file_utils.touch)
    return result_rdd


class MultiCalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'multi_calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated'

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(get_vehicle_type)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type[0])
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type[0])), 3)

        """ get to do jobs """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs', get_todo_dirs(origin_vehicle_dir), 3)

        """ get conf files """
        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # PairRDD(vehicle, dir_of_vehicle)
            origin_vehicle_dir
            # PairRDD(vehicle_type, vehicle_conf)
            .mapValues(get_vehicle_param), 3)

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run_prod(self):
        origin_prefix = 'modules/control/data/records'
        target_prefix = 'modules/control/data/results'

        """ get conf files """
        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file', get_vehicle_param_prod(origin_prefix))

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(get_vehicle_type_prod)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type)))

        """ get to do jobs """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            # PairRDD(vehicle_type, path_to_vehicle_type)
            origin_vehicle_dir
            # PairRDD(vehicle_type, files)
            .flatMapValues(list_end_files_prod)
            # PairRDD(vehicle_type, 'COMPLETE'_files)
            .filter(lambda key_path: key_path[1].endswith('COMPLETE'))
            # PairRDD(vehicle_type, path_to_'COMPLETE')
            .mapValues(os.path.dirname))

        processed_dirs = spark_helper.cache_and_log(
            'processed_jobs',
            # PairRDD(vehicle, abs_task_dir)
            todo_task_dirs
            # PairRDD(vehicle, task_dir_with_target_prefix)
            .map(lambda (vehicle, path):
                 (vehicle, path.replace(os.path.join(origin_prefix, vehicle),
                                        os.path.join(target_prefix, 'CalibrationTableFeature',
                                                     vehicle, 'throttle', 'train'), 1)))
            # PairRDD(vehicle, relative_task_dir)
            .mapValues(get_relative_path)
            # PairRDD(vehicle, files)
            .flatMapValues(list_end_files_prod)
            # PairRDD(vehicle, file_end_with_MARKER)
            .filter(lambda key_path: key_path[1].endswith(MARKER))
            # PairRDD(vehicle, file_end_with_MARKER with origin prefix)
            .map(lambda (vehicle, path):
                 (vehicle, path.replace(os.path.join(target_prefix, 'CalibrationTableFeature',
                                                     vehicle, 'throttle', 'train'),
                                        os.path.join(origin_prefix, vehicle), 1)))
            # PairRDD(vehicle, dir of file_end_with_MARKER with origin prefix)
            .mapValues(os.path.dirname))

        # PairRDD(vehicle_type, dir_of_todos_with_origin_prefix)
        todo_task_dirs = todo_task_dirs.subtract(processed_dirs)

        self.run(todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix)

    def run(self, todo_task_dirs, vehicle_param_conf, origin_prefix, target_prefix):
        records = spark_helper.cache_and_log(
            'Records',
            todo_task_dirs
            # PairRDD(vehicle, files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*')))
            # PairRDD(vehicle, records)
            .filter(lambda (_, end_file): record_utils.is_record_file(end_file))
            # PairRDD(vehicle, (dir, records))
            .mapValues(lambda records: (os.path.dirname(records), records))
            # PairRDD((vehicle, dir), records)
            .map(lambda (vehicle, (record_dir, records)): ((vehicle, record_dir), records)))

        # PairRDD((vehicle, segment_dir, segment_id), msg)
        valid_msg_segments = spark_helper.cache_and_log('Valid_msgs', valid_segment(records))

        parsed_msgs = spark_helper.cache_and_log(
            'parsed_msg',
            # PairRDD((vehicle, dir, segment_id), (chassis_msgs, pose_msgs))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msg_segments)
            # PairRDD((vehicle, dir, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(pair_cs_pose))

        msgs_with_conf = spark_helper.cache_and_log(
            'msgs_with_conf',
            # PairRDD(vehicle, (segment_dir, segment_id, paired_chassis_msg_pose_msg))
            parsed_msgs.map(lambda ((vehicle, segment_dir, segment_id), msgs):
                            (vehicle, (segment_dir, segment_id, msgs)))
            # PairRDD(vehicle,
            #         ((dir_segment, segment_id, paired_chassis_msg_pose_msg), vehicle_param_conf))
            .join(vehicle_param_conf)
            # PairRDD((vehicle, dir_segment, segment_id),
            #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
            .map(lambda (vehicle, ((segment_dir, segment_id, msgs), conf)):
                 ((vehicle, segment_dir, segment_id), (msgs, conf))))

        data_rdd = spark_helper.cache_and_log(
            'data_rdd',
            # PairRDD((vehicle, dir_segment, segment_id),
            #         (paired_chassis_msg_pose_msg, vehicle_param_conf))
            msgs_with_conf
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (msgs, conf):
                       (calibration_table_utils.feature_generate(msgs, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_filter(features), conf))
            # PairRDD((vehicle, dir_segment, segment_id), (features, vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_cut(features, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id),
            #         ((grid_dict, features), vehicle_param_conf))
            .mapValues(lambda (features, conf):
                       (calibration_table_utils.feature_distribute(features, conf), conf))
            # PairRDD((vehicle, dir_segment, segment_id), feature_data_matrix)
            .mapValues(lambda (features_grid, conf):
                       calibration_table_utils.feature_store(features_grid, conf)))

        # write data to hdf5 files
        result_rdd = spark_helper.cache_and_log(
            'result_rdd',
            # PairRDD((vehicle, dir_segment, segment_id), feature_data_matrix)
            data_rdd
            # RDD(feature_numbers)
            .map(lambda elem: write_h5(elem, origin_prefix, target_prefix)))

        # RDD (dir_segment)
        complete_rdd = spark_helper.cache_and_log(
            'completed_dirs',
            mark_complete(valid_msg_segments, origin_prefix,
                          target_prefix, MARKER))


if __name__ == '__main__':
    MultiCalibrationTableFeatureExtraction().main()
