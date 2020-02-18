#!/usr/bin/env python
"""
common functions for feature extractin
"""
import os
import glob

import h5py
import math
import numpy as np

from modules.canbus.proto.chassis_pb2 import Chassis
from modules.common.configs.proto import vehicle_config_pb2
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
from modules.localization.proto.localization_pb2 import LocalizationEstimate

from fueling.control.dynamic_model.conf.model_config import point_mass_config
from fueling.control.proto.feature_key_pb2 import FeatureKey
import fueling.common.file_utils as file_utils
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils
import fueling.control.dynamic_model.conf.model_config as model_config

VEHICLE_MODEL = point_mass_config["vehicle_model"]

FILENAME_FEATURE_KEY_CONF = file_utils.data_path("fueling/control/conf/feature_key_conf.pb.txt")
FEATURE_KEY = proto_utils.get_pb_from_text_file(FILENAME_FEATURE_KEY_CONF, FeatureKey())

# vehicle param constant
FILENAME_VEHICLE_PARAM_CONF = os.path.join(
    '/apollo/modules/calibration/data', VEHICLE_MODEL, 'vehicle_param.pb.txt')
VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(FILENAME_VEHICLE_PARAM_CONF,
                                                       vehicle_config_pb2.VehicleConfig())

THROTTLE_DEADZONE = VEHICLE_PARAM_CONF.vehicle_param.throttle_deadzone
THROTTLE_MAX = FEATURE_KEY.throttle_max

BRAKE_DEADZONE = VEHICLE_PARAM_CONF.vehicle_param.brake_deadzone
BRAKE_MAX = FEATURE_KEY.brake_max

SPEED_STEP = FEATURE_KEY.speed_step
SPEED_MAX = FEATURE_KEY.speed_max

ACC_STEP = FEATURE_KEY.acc_step  # percentage
STEER_STEP = FEATURE_KEY.steer_step  # percentage

WANTED_VEHICLE = FEATURE_KEY.vehicle_type

MAX_PHASE_DELTA_SEGMENT = 0.015
MAX_PHASE_DELTA = 0.005
MIN_SEGMENT_LENGTH = 10


def get_vehicle_of_dirs(dir_to_records_rdd):
    """
    Extract HMIStatus.current_vehicle from each dir.
    Convert RDD(dir, record) to RDD(dir, vehicle).
    """
    logging.info('records: {}'.format(dir_to_records_rdd))

    def _get_vehicle_from_records(records):
        reader = record_utils.read_record([record_utils.HMI_STATUS_CHANNEL])
        logging.info('records: {}'.format(records))

        for record in records:
            logging.info('Try getting vehicle name from {}'.format(record))
            for msg in reader(record):
                hmi_status = record_utils.message_to_proto(msg)
                vehicle = hmi_status.current_vehicle
                logging.info('Get vehicle name "{}" from record {}'.format(vehicle, record))
                return vehicle
        logging.info('Failed to get vehicle name')
        return ''
    return dir_to_records_rdd.groupByKey().mapValues(_get_vehicle_from_records)


def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    task_dir, msg = dir_to_msg
    dt = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = dt.strftime('%Y%m%d-%H%M')
    return ((task_dir, segment_id), msg)


def gen_key(elem):
    """ generate a key contains both folder path and timestamp """
    return ((elem[0], (int)(elem[1].header.timestamp_sec / 60)), elem[1])


def process_seg(elem):
    """group Chassis and Localization msgs to list separately"""
    chassis = []
    pose = []
    for each_elem in elem:
        if each_elem.header.module_name == "canbus":
            chassis.append(each_elem)
        else:
            pose.append(each_elem)
    return (chassis, pose)


def pair_cs_pose(elem):
    """pair chassis and pose"""
    chassis = elem[0]
    pose = elem[1]
    chassis.sort(key=lambda x: x.header.timestamp_sec)
    pose.sort(key=lambda x: x.header.timestamp_sec)
    times_pose = np.array([x.header.timestamp_sec for x in pose])
    times_cs = np.array([x.header.timestamp_sec for x in chassis])
    index = [0, 0]
    res = []

    while index[0] < len(times_cs) and index[1] < len(times_pose):
        if abs(times_cs[index[0]] - times_pose[index[1]]) <= MAX_PHASE_DELTA:
            res.append((chassis[index[0]], pose[index[1]]))
            index[0] += 1
            index[1] += 1
        elif times_cs[index[0]] < times_pose[index[1]]:
            index[0] += 1
        else:
            index[1] += 1
    return res


def gen_data_point(pose, chassis):
    return np.array([
        pose.heading,  # 0
        pose.orientation.qx,  # 1
        pose.orientation.qy,  # 2
        pose.orientation.qz,  # 3
        pose.orientation.qw,  # 4
        pose.linear_velocity.x,  # 5
        pose.linear_velocity.y,  # 6
        pose.linear_velocity.z,  # 7
        pose.linear_acceleration.x,  # 8
        pose.linear_acceleration.y,  # 9
        pose.linear_acceleration.z,  # 10
        pose.angular_velocity.x,  # 11
        pose.angular_velocity.y,  # 12
        pose.angular_velocity.z,  # 13
        chassis.speed_mps,  # 14 speed
        chassis.throttle_percentage / 100,  # 15 throttle
        chassis.brake_percentage / 100,  # 16 brake
        chassis.steering_percentage / 100,  # 17
        chassis.driving_mode,  # 18
        pose.position.x,  # 19
        pose.position.y,  # 20
        pose.position.z,  # 21
        chassis.gear_location,  # 22
    ])


def multi_get_data_point(elem):
    """ extract data from msg """
    (vehicle, data_dir, timestamp_sec), (chassis, pose_pre) = elem
    pose = pose_pre.pose
    return (vehicle, (data_dir, chassis.header.timestamp_sec, gen_data_point(pose, chassis)))


def get_data_point(elem):
    """ extract data from msg """
    (data_dir, timestamp_sec), (chassis, pose_pre) = elem
    pose = pose_pre.pose
    return ((data_dir, chassis.header.timestamp_sec), gen_data_point(pose, chassis))


def gen_steering_key(steering):
    if steering < -60.0:  # right
        return 0
    elif -60.0 <= steering < -30.0:
        return 1
    elif -30.0 <= steering < -1.0:
        return 2
    elif -1.0 <= steering < 1.0:
        return 3
    elif 1.0 <= steering < 30:
        return 4
    elif 30 <= steering < 60:
        return 5
    else:
        return 6


def gen_speed_key(non_stop_speed):
    if non_stop_speed < 10:
        return 1
    elif 10 <= non_stop_speed < 20:
        return 2
    else:
        return 3


def gen_brake_key(brake, brake_deadzone=BRAKE_DEADZONE):
    if brake < BRAKE_DEADZONE:
        return 0
    elif BRAKE_DEADZONE <= brake < 20:
        return 1
    elif 20 <= brake < 25:
        return 2
    else:
        return 3


def gen_throttle_key(throttle, throttle_deadzone=THROTTLE_DEADZONE):
    if throttle < THROTTLE_DEADZONE:
        return 0
    elif THROTTLE_DEADZONE <= throttle < 25:
        return 1
    elif 25 <= throttle < 30:
        return 2
    else:
        return 3


def gen_reverse_throttle_key(throttle, throttle_deadzone=THROTTLE_DEADZONE):
    if throttle < THROTTLE_DEADZONE:
        return 0
    elif THROTTLE_DEADZONE <= throttle < 20:
        return 1
    else:
        return 2


def multi_gen_feature_key(elem):
    """ generate label for both forward driving"""
    (vehicle, data_dir, time_stamp), (data_point, conf) = elem
    speed = data_point[14]
    throttle = data_point[15] * 100  # 0 or positive
    brake = data_point[16] * 100  # 0 or positive
    steering = data_point[17] * 100
    gear = data_point[22]
    gear_key = int(gear)

    # forward driving:
    if gear_key != 1 and speed <= 0:  # check if it backward driving
        elem_key = int(10000)
    elif speed < conf.max_abs_speed_when_stopped:
        elem_key = int(9000)
    else:
        steering_key = int(gen_steering_key(steering))
        brake_key = int(gen_brake_key(brake, conf.brake_deadzone))
        speed_key = int(gen_speed_key(speed))
        throttle_key = int(gen_throttle_key(throttle, conf.throttle_deadzone))
        elem_key = int(speed_key * 1000 + steering_key * 100 + throttle_key * 10 + brake_key)
    # ((vehicle, path, feature_key), (time_stamp, paired_data))
    return ((vehicle, data_dir, elem_key), (time_stamp, data_point))


def multi_gen_feature_key_backwards(elem):
    """ generate label for backward driving"""
    (vehicle, data_dir, time_stamp), (data_point, conf) = elem
    speed = data_point[14]
    throttle = data_point[15] * 100  # 0 or positive
    brake = data_point[16] * 100  # 0 or positive
    steering = data_point[17] * 100
    gear = data_point[22]
    gear_key = int(gear)

    if gear_key != 2 and speed <= 0:  # check if it fardward driving
        elem_key = int(10000)
    elif speed < conf.max_abs_speed_when_stopped:
        elem_key = int(9000)
    else:
        steering_key = int(gen_steering_key(steering))
        throttle_key = int(gen_throttle_key(throttle, conf.throttle_deadzone))
        brake_key = int(gen_brake_key(brake, conf.brake_deadzone))
        # steering-throttle-brake
        elem_key = int(steering_key * 100 + throttle_key * 10 + brake_key)

    # ((folder_path, feature_key), (time_stamp, paired_data))
    return ((vehicle, data_dir, elem_key), (time_stamp, data_point))


def gen_feature_key(elem):
    """ generate label for both forward driving"""
    speed = elem[1][14]
    throttle = elem[1][15] * 100  # 0 or positive
    brake = elem[1][16] * 100  # 0 or positive
    steering = elem[1][17] * 100
    gear = elem[1][22]
    gear_key = int(gear)

    # forward driving:
    if gear_key != 1:  # check if it backward driving
        elem_key = int(10000)
    elif speed < VEHICLE_PARAM_CONF.vehicle_param.max_abs_speed_when_stopped:
        elem_key = int(9000)
    else:
        steering_key = int(gen_steering_key(steering))
        brake_key = int(gen_brake_key(brake))
        speed_key = int(gen_speed_key(speed))
        throttle_key = int(gen_throttle_key(throttle))
        elem_key = int(speed_key * 1000 + steering_key * 100 + throttle_key * 10 + brake_key)
    # ((folder_path, feature_key), (time_stamp, paired_data))
    return ((elem[0][0], elem_key), (elem[0][1], elem[1]))


def gen_feature_key_backwards(elem):
    """ generate label for backward driving"""
    speed = elem[1][14]
    throttle = elem[1][15] * 100  # 0 or positive
    brake = elem[1][16] * 100  # 0 or positive
    steering = elem[1][17] * 100
    gear = elem[1][22]
    gear_key = int(gear)
    # logging.info('gear: %d' % gear)

    if gear_key != 2:  # check if it fardward driving
        elem_key = int(10000)
    elif speed < VEHICLE_PARAM_CONF.vehicle_param.max_abs_speed_when_stopped:
        elem_key = int(9000)
    else:
        steering_key = int(gen_steering_key(steering))
        throttle_key = int(gen_throttle_key(throttle))
        brake_key = int(gen_brake_key(brake))
        # speed-steering-throttle-brake
        elem_key = int(steering_key * 100 + throttle_key * 10 + brake_key)

    # ((folder_path, feature_key), (time_stamp, paired_data))
    return ((elem[0][0], elem_key), (elem[0][1], elem[1]))


def gen_segment(elem):
    """ generate segment w.r.t time """
    segments = []
    pre_time = elem[0][0]
    data_set = np.array(elem[0][1])
    counter = 1  # count segment length first element
    for i in range(1, len(elem)):
        # print('len of elem: %d', i)
        if (elem[i][0] - pre_time) <= MAX_PHASE_DELTA_SEGMENT:
            data_set = np.vstack([data_set, elem[i][1]])
            counter += 1
        else:
            # logging.info('time differences: %f' % (elem[i][0] - pre_time))
            if counter > model_config.feature_config['sequence_length']:
                segments.append((segment_id(pre_time), data_set))
            data_set = np.array([elem[i][1]])
            counter = 0
        pre_time = elem[i][0]
        # logging.info('previous time: %f' % pre_time)
    if counter > model_config.feature_config['sequence_length']:
        segments.append((segment_id(pre_time), data_set))
    return segments


def segment_id(timestamp):
    return int(timestamp * 100) % 1000000


def write_segment_with_key(elem, origin_prefix, target_prefix):
    """write to h5 file, use feature key as file name"""
    ((folder_path, key), (segmentID, data_set)) = elem
    folder_path = folder_path.replace(origin_prefix, target_prefix, 1)
    file_name = str(key) + '_' + str(segmentID)
    h5_utils.write_h5_single_segment(data_set, folder_path, file_name)
    return (folder_path, key)


def multi_write_segment_with_key(elem, origin_prefix, target_prefix):
    """write to h5 file, use feature key as file name"""
    ((vehicle, folder_path, key), (segmentID, data_set)) = elem
    folder_path = folder_path.replace(origin_prefix, target_prefix, 1)
    file_name = str(key) + '_' + str(segmentID)
    h5_utils.write_h5_single_segment(data_set, folder_path, file_name)
    return (folder_path, key)
