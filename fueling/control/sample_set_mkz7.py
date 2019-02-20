#!/usr/bin/env python
""" extracting sample set for mkz7 """
# pylint: disable = fixme
# pylint: disable = no-member
import glob

import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

import fueling.common.record_utils as record_utils
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
import modules.control.proto.control_conf_pb2 as ControlConf
import modules.data.fuel.fueling.control.proto.feature_key_pb2 as FeatureKey
import common.proto_utils as proto_utils


HMI_CH = ['/apollo/hmi/status']

WANTED_VEHICLE = 'Mkz7'

PATH = ["/apollo/modules/data/fuel/fueling/control/records/right-40-10",
        "/apollo/modules/data/fuel/fueling/control/records/left-40-10",
        "/apollo/modules/data/fuel/fueling/control/records/Transient_1"]

WANTED_CHS = ['/apollo/canbus/chassis',
              '/apollo/localization/pose']

MAX_PHASE_DELTA = 0.01
MIN_SEGMENT_LENGTH = 10

FEATURE_KEY = FeatureKey.featureKey()
FILENAME = "/apollo/modules/data/fuel/fueling/control/conf/feature_key_conf.pb.txt"
proto_utils.get_pb_from_text_file(FILENAME, FEATURE_KEY)

CONTROL_CONF = ControlConf.ControlConf()
FILENAME_CONTROL_CONF = "/apollo/modules/control/conf/control_conf.pb.txt"
proto_utils.get_pb_from_text_file(FILENAME_CONTROL_CONF, CONTROL_CONF)


# TODO change based on vehicle model
THROTTLE_DEADZONE = CONTROL_CONF.lon_controller_conf.throttle_deadzone
THROTTLE_MAX = FEATURE_KEY.throttle_max

BRAKE_DEADZONE = CONTROL_CONF.lon_controller_conf.brake_deadzone
BRAKE_MAX = FEATURE_KEY.brake_max

SPEED_SLICE = FEATURE_KEY.speed_slice
SPEED_MAX = FEATURE_KEY.speed_max

ACC_SLICE = FEATURE_KEY.acc_slice  # percentage
STEER_SLICE = FEATURE_KEY.steer_slice  # percentage


def folder_to_record(pathname):
    """ folder path to record path"""
    return glob.glob(pathname + "/*.record.*")


def folder_to_record_with_key(pathname):
    """ folder path to record path"""
    return glob.glob(pathname + "/*.record.*")


def process_hmi_msg(msg):
    """Parse message"""
    msg_new = HMIStatus()
    msg_new.ParseFromString(msg.message)
    return msg_new


def current_vehicle(elem):
    """ extract vehicle type """
    return elem.current_vehicle


def is_wanted_vehicle(elem):
    """ check if it is wantted vehicle """
    return elem[1] == WANTED_VEHICLE


def to_list(elem):
    """convert element to list"""
    return [(elem)]


def append(orig_elem, app_elem):
    """append another element to previous element"""
    orig_elem.append((app_elem))
    return orig_elem


def extend(orig_elem, app_elem):
    """extend the original list"""
    orig_elem.extend(app_elem)
    return orig_elem


def process_msg(msg):
    """Parse message"""
    if msg.topic == "/apollo/canbus/chassis":
        msg_new = Chassis()
    else:
        msg_new = LocalizationEstimate()
    msg_new.ParseFromString(msg.message)
    return msg_new


def gen_folder_time_key(elem):
    """ generate a key contians both folder path and time stamp """
    return ((elem[0], (int)(elem[1].header.timestamp_sec/60)), elem[1])


def process_seg(elems):
    """group Chassis and Localization msgs to list seperately"""
    chassis = []
    pose = []
    for elem in elems:
        if elem.header.module_name == "canbus":
            chassis.append(elem)
        else:
            pose.append(elem)
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
        if abs(times_cs[index[0]] - times_pose[index[1]]) < MAX_PHASE_DELTA:
            res.append((chassis[index[0]], pose[index[1]]))
            index[0] += 1
            index[1] += 1
        else:
            while index[0] < len(times_cs) and index[1] < len(times_pose)\
                    and times_cs[index[0]] < times_pose[index[1]] - MAX_PHASE_DELTA:
                index[0] += 1
            while index[0] < len(times_cs) and index[1] < len(times_pose)\
                    and times_pose[index[1]] < times_cs[index[0]] - MAX_PHASE_DELTA:
                index[1] += 1

    return res


def get_data_point(elem):
    """ extract data from msg """
    chassis = elem[1][0]
    pose = elem[1][1].pose
    res = np.array([
        pose.heading - 0.11,  # 0
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


# extract folders for specific vehicle type
FOLDERS_VEHICLE = (spark_helper.get_context('Test')
                   # folder path
                   .parallelize(PATH)
                   # key as folder path
                   .keyBy(lambda x: x)
                   .flatMapValues(folder_to_record)
                   .flatMapValues(record_utils.read_record(HMI_CH))
                   # parse message
                   .mapValues(process_hmi_msg)
                   .mapValues(current_vehicle)
                   .filter(is_wanted_vehicle)
                   # remove duplication of folders
                   .distinct()
                   # choose only folder path
                   .map(lambda x: x[0]))

print FOLDERS_VEHICLE.first()


CHANNELS = (FOLDERS_VEHICLE
            .keyBy(lambda x: x)
            # record path
            .flatMapValues(folder_to_record)
            # read message
            .flatMapValues(record_utils.read_record(WANTED_CHS))
            # parse message
            .mapValues(process_msg))

print CHANNELS.count()

PRE_SEGMENT = (CHANNELS
               # choose time as key, group msg into 1 sec
               .map(gen_folder_time_key)
               # combine chassis message and pose message with the same key
               .combineByKey(to_list, append, extend))

print PRE_SEGMENT.count()

DATA = (PRE_SEGMENT
        # msg list
        .mapValues(process_seg)
        # flat value to paired data points
        .flatMapValues(pair_cs_pose)
        .map(get_data_point))
print DATA.count()

# data feature set
FEATURED_DATA = (DATA
                 .map(feature_key_value)
                 .combineByKey(to_list, append, extend))

print FEATURED_DATA.count()


def gen_segment(elem):
    """ generate segment w.r.t time """
    segments = []
    pre_time = elem[0][0]
    data_set = np.array(elem[0][1])
    for i in range(1, len(elem)):
        if (elem[i][0]-pre_time) <= 2*MAX_PHASE_DELTA:
            data_set = np.vstack([data_set, elem[i][1]])
        else:
            if i > MIN_SEGMENT_LENGTH:
                segments.append(data_set)
            data_set = np.array([elem[i][1]])
        pre_time = elem[i][0]
    segments.append(data_set)
    return segments


def write_h5(elem):
    """write to h5 file, use feature key as file name"""
    key = str(elem[0][1])
    folder_path = str(elem[0][0])
    out_file = h5py.File(
        "{}/training_dataset_{}.hdf5".format(folder_path, key), "w")
    i = 0
    for data_set in elem[1]:
        name = "_segment_" + str(i).zfill(3)
        out_file.create_dataset(name, data=data_set, dtype="float32")
        i += 1
    out_file.close()
    return elem[0]


DATA_SEGMENT = (FEATURED_DATA
                # generate segment w.r.t time
                .mapValues(gen_segment)
                # write all segment into a hdf5 file
                .map(write_h5))
print DATA_SEGMENT.first()
