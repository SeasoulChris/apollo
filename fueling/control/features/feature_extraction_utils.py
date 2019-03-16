#!/usr/bin/env python
"""
common functions for feature extractin
"""
import os
import glob

import glog
import h5py
import numpy as np

from modules.canbus.proto.chassis_pb2 import Chassis
from modules.control.proto.control_conf_pb2 import ControlConf
from modules.data.fuel.fueling.control.proto.feature_key_pb2 import featureKey
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
from modules.localization.proto.localization_pb2 import LocalizationEstimate
import common.proto_utils as proto_utils
import fueling.common.h5_utils as h5_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils


FILENAME = "/mnt/bos/modules/control/common/feature_key_conf.pb.txt"
FEATURE_KEY = proto_utils.get_pb_from_text_file(FILENAME, featureKey())

FILENAME_CONTROL_CONF = "/mnt/bos/modules/control/common/control_conf.pb.txt"
CONTROL_CONF = proto_utils.get_pb_from_text_file(
    FILENAME_CONTROL_CONF, ControlConf())

# TODO change based on vehicle model
THROTTLE_DEADZONE = 5.0  # CONTROL_CONF.lon_controller_conf.throttle_deadzone
THROTTLE_MAX = FEATURE_KEY.throttle_max

BRAKE_DEADZONE = 7.0  # CONTROL_CONF.lon_controller_conf.brake_deadzone
BRAKE_MAX = FEATURE_KEY.brake_max

SPEED_SLICE = FEATURE_KEY.speed_slice
SPEED_MAX = FEATURE_KEY.speed_max

ACC_SLICE = FEATURE_KEY.acc_slice  # percentage
STEER_SLICE = FEATURE_KEY.steer_slice  # percentage

MAX_PHASE_DELTA = 0.01
MIN_SEGMENT_LENGTH = 10


def get_vehicle_of_dirs(dir_to_records_rdd):
    """
    Extract HMIStatus.current_vehicle from each dir.
    Convert RDD(dir, record) to RDD(dir, vehicle).
    """
    def _get_vehicle_from_records(records):
        reader = record_utils.read_record([record_utils.HMI_STATUS_CHANNEL])
        for record in records:
            glog.info('Try getting vehicle name from {}'.format(record))
            for msg in reader(record):
                hmi_status = record_utils.message_to_proto(msg)
                vehicle = hmi_status.current_vehicle
                glog.info('Get vehicle name "{}" from record {}'.format(
                    vehicle, record))
                return vehicle
        glog.info('Failed to get vehicle name')
        return ''
    return dir_to_records_rdd.groupByKey().mapValues(_get_vehicle_from_records)


def gen_pre_segment(dir_to_msg):
    """Generate new key which contains a segment id part."""
    task_dir, msg = dir_to_msg
    dt = time_utils.msg_time_to_datetime(msg.timestamp)
    segment_id = dt.strftime('%Y%m%d-%H%M')
    return ((task_dir, segment_id), msg)


def gen_key(elem):
    """ generate a key contians both folder path and time stamp """
    return ((elem[0], (int)(elem[1].header.timestamp_sec/60)), elem[1])


def process_seg(elem):
    """group Chassis and Localization msgs to list seperately"""
    chassis = []
    pose = []
    for each_elem in elem:
        if each_elem.header.module_name == "canbus":
            chassis.append(each_elem)
        else:
            pose.append(each_elem)
    return (chassis, pose)


def to_list(elem):
    """convert element to list"""
    return [elem]


def append(orig_elem, app_elem):
    """append another element to the revious element"""
    orig_elem.append((app_elem))
    return orig_elem


def extend(orig_elem, app_elem):
    """extend the original list"""
    orig_elem.extend(app_elem)
    return orig_elem


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
        if abs(times_cs[index[0]] - times_pose[index[1]]) < MAX_PHASE_DELTA:
            res.append((chassis[index[0]], pose[index[1]]))
            index[0] += 1
            index[1] += 1
        else:
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_cs[index[0]] < times_pose[index[1]] - MAX_PHASE_DELTA:
                index[0] += 1
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_pose[index[1]] < times_cs[index[0]] - MAX_PHASE_DELTA:
                index[1] += 1

    return res


def get_data_point(elem):
    """ extract data from msg """
    chassis = elem[1][0]
    pose = elem[1][1].pose
    res = np.array([
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
        chassis.speed_mps,  # 14
        chassis.throttle_percentage/100,  # 15
        chassis.brake_percentage/100,  # 16
        chassis.steering_percentage/100,  # 17
        chassis.driving_mode,  # 18
        pose.position.x,  # 19
        pose.position.y,  # 20
        pose.position.z,  # 21
    ])
    return ((elem[0][0], chassis.header.timestamp_sec), res)


def feature_key_value(elem):
    """ generate key for each data segment """
    speed = elem[1][14]
    throttle = max(elem[1][15]*100 - THROTTLE_DEADZONE, 0)  # 0 or positive
    brake = max(elem[1][16]*100-BRAKE_DEADZONE, 0)  # 0 or positive
    steering = elem[1][17]*100+100  # compensation for negative value

    # speed key 1 ~ 4
    speed_key = int(min(speed, SPEED_MAX)/SPEED_SLICE+1)

    # steering key 0 ~ 9
    steering_key = int(steering/STEER_SLICE)  # -100% ~ 0

    throttle_key = int(min(throttle, THROTTLE_MAX)/ACC_SLICE)
    brake_key = int(min(brake, BRAKE_MAX)/ACC_SLICE)

    elem_key = int(speed_key*1000+steering_key *
                   100 + throttle_key*10+brake_key)
    # ((folder_path,feature_key),(time_stamp,paired_data))
    return ((elem[0][0], elem_key), (elem[0][1], elem[1]))


def gen_segment(elem):
    """ generate segment w.r.t time """
    segments = []
    pre_time = elem[0][0]
    data_set = np.array(elem[0][1])
    for i in range(1, len(elem)):
        if (elem[i][0] - pre_time) <= 2 * MAX_PHASE_DELTA:
            data_set = np.vstack([data_set, elem[i][1]])
        else:
            if i > MIN_SEGMENT_LENGTH:
                segments.append(data_set)
            data_set = np.array([elem[i][1]])
        pre_time = elem[i][0]
    segments.append(data_set)
    return segments


def write_h5_with_key(elem, origin_prefix, target_prefix, vehicle_type):
    """write to h5 file, use feature key as file name"""
    key = str(elem[0][1])
    folder_path = str(elem[0][0])
    folder_path = folder_path.replace(origin_prefix, target_prefix, 1)
    file_name = vehicle_type+'_'+key
    h5_utils.write_h5(elem[1], folder_path, file_name)
    return elem[0]
