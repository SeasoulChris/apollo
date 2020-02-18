#!/usr/bin/env python
from collections import defaultdict
import copy
import glob
import math
import os
import random

import h5py
import numpy as np

from modules.common.configs.proto.vehicle_config_pb2 import VehicleParam

from fueling.control.features.filters import Filters
from fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.file_utils as file_utils
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils


calibration_conf_file = file_utils.data_path('fueling/control/conf/calibration_table_conf.pb.txt')
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(
    calibration_conf_file, CalibrationTable())


segment_store_num = 12
wanted_driving_mode = "COMPLETE_MANUAL"


def get_conf_value(msgs):
    # get max value from data
    speed_max = 0.0
    throttle_max = 0.0  # positive value
    brake_max = 0.0  # positive value
    speed_min = 30.0
    acc_min = 0.0  # negative
    acc_max = 0.0
    for msg in msgs:
        chassis, pose_pre = msg
        pose = pose_pre.pose
        heading_angle = pose.heading
        acc_x = pose.linear_acceleration.x
        acc_y = pose.linear_acceleration.y
        acc = acc_x * math.cos(heading_angle) + acc_y * math.sin(heading_angle)
        if int(chassis.gear_location) != 1 or chassis.speed_mps < 0:  # keep only gear_drive data
            # logging.info("chassis.gear_location %s" % chassis.gear_location)
            continue
        throttle_max = max(chassis.throttle_percentage, throttle_max)
        brake_max = max(chassis.brake_percentage, brake_max)
        speed_max = max(chassis.speed_mps, speed_max)
        speed_min = min(chassis.speed_mps, speed_min)
        acc_min = min(acc, acc_min)
        acc_max = max(acc, acc_max)
    return (speed_min, speed_max, throttle_max, brake_max, acc_min, acc_max)


def compare_conf_value(conf_value_x, conf_value_y):
    if not conf_value_x:
        return conf_value_y
    elif not conf_value_y:
        return conf_value_x
    speed_min_x, speed_max_x, throttle_max_x, brake_max_x, acc_min_x, acc_max_x = conf_value_x
    speed_min_y, speed_max_y, throttle_max_y, brake_max_y, acc_min_y, acc_max_y = conf_value_y
    speed_min = min(speed_min_x, speed_min_y)
    speed_max = max(speed_max_x, speed_max_y)
    throttle_max = max(throttle_max_x, throttle_max_y)
    brake_max = max(brake_max_x, brake_max_y)
    acc_max = max(acc_max_x, acc_max_y)
    acc_min = min(acc_min_x, acc_min_y)
    return (speed_min, speed_max, throttle_max, brake_max, acc_min, acc_max)


def write_conf(conf_value, vehicle_param_conf, train_conf_path, train_conf=CALIBRATION_TABLE_CONF):
    cur_conf = copy.deepcopy(train_conf)
    speed_min, speed_max, throttle_max, brake_max, acc_min, acc_max = conf_value

    cur_conf.speed_min = max(speed_min, vehicle_param_conf.max_abs_speed_when_stopped)
    cur_conf.speed_max = min(speed_max, train_conf.speed_max)
    cur_conf.throttle_max = min(throttle_max, train_conf.throttle_max)
    cur_conf.brake_max = min(brake_max, train_conf.brake_max)
    # acc range is a supper set of the range in vehicle_param
    cur_conf.acc_min = min(acc_min, vehicle_param_conf.max_deceleration)
    cur_conf.acc_max = max(acc_max, vehicle_param_conf.max_deceleration)

    logging.info('Load calibration table conf: %s' % cur_conf)
    file_utils.makedirs(train_conf_path)
    with open(os.path.join(train_conf_path, 'calibration_table_conf.pb.txt'), 'w') as fin:
        fin.write(str(cur_conf))
    return 0


def decide_cmd(chassis_throttle_val, chassis_brake_val, vehicle_param_conf):
    """decide cmd is brake or throttle"""
    break_deadzone = vehicle_param_conf.brake_deadzone
    throttle_deadzone = vehicle_param_conf.throttle_deadzone
    feature_cmd = 0.0
    if chassis_throttle_val > abs(throttle_deadzone):
        feature_cmd = chassis_throttle_val
    elif chassis_brake_val > abs(break_deadzone):
        feature_cmd = -chassis_brake_val
    else:
        feature_cmd = 0.0
    return feature_cmd


def feature_generate(elems, vehicle_param_conf):
    """ extract data from segment """
    res = np.zeros([len(elems), 5])
    counter = 0
    for elem in elems:
        chassis = elem[0]
        pose = elem[1].pose

       # check gear
        if int(chassis.gear_location) != 1 or chassis.speed_mps < 0.0:  # keep only gear_drive data
            continue
        heading_angle = pose.heading

        acc_x = pose.linear_acceleration.x
        acc_y = pose.linear_acceleration.y
        acc = acc_x * math.cos(heading_angle) + acc_y * math.sin(heading_angle)
        # throttle cmd is positive
        # brake cmd is negative
        feature_cmd = decide_cmd(chassis.throttle_percentage,
                                 chassis.brake_percentage, vehicle_param_conf)

        driving_mode = (chassis.driving_mode == wanted_driving_mode)

        res[counter] = np.array([
            chassis.speed_mps,  # 0: speed
            acc,  # 1: acc
            feature_cmd,                # 2: cmd
            chassis.steering_percentage,
            driving_mode
        ])
        counter += 1
    return res


def gen_cmd_list(vehicle_param_conf, train_conf):
    segment_brake_list = np.linspace(-1 * train_conf.brake_max, -1 * vehicle_param_conf.brake_deadzone,
                                     num=train_conf.brake_segment).tolist()
    segment_throttle_list = np.linspace(
        vehicle_param_conf.throttle_deadzone,
        train_conf.throttle_max,
        num=train_conf.throttle_segment).tolist()
    return segment_brake_list + segment_throttle_list


def satisfy_brake_condition(elem, index, vehicle_param_conf, train_conf):
    """
    whether satisfy brake condition
    """
    brake_max_condition = -1 * vehicle_param_conf.brake_deadzone
    brake_min_condition = -1 * train_conf.brake_max
    # vehicle param conf is not vehicle actual acc limit
    # acc_min_condition = vehicle_param_conf.max_deceleration
    steer_condition = train_conf.steer_condition
    condition = (
        abs(elem[index][3]) < steer_condition and
        brake_min_condition < elem[index][2] < brake_max_condition and
        elem[index][1] < 0.0 and
        int(elem[index][4]) == 0)
    return condition


def satisfy_throttle_condition(elem, index, vehicle_param_conf, train_conf):
    """
    whether satisfy throttle condition
    """
    throttle_min_condition = vehicle_param_conf.throttle_deadzone
    throttle_max_condition = train_conf.throttle_max
    # vehicle param conf is not vehicle actual acc limit
    # acc_max_condition = vehicle_param_conf.max_acceleration
    steer_condition = train_conf.steer_condition
    condition = (
        abs(elem[index][3]) < steer_condition and
        throttle_min_condition < elem[index][2] < throttle_max_condition and
        0.0 < elem[index][1] and
        int(elem[index][4]) == 0)
    return condition


def feature_cut(elem, vehicle_param_conf, train_conf):
    """
    get desired feature interval
    """
    id_elem = 0
    num_row = elem.shape[0]
    # find satisfied data
    for i in range(num_row):
        if (satisfy_throttle_condition(elem, i, vehicle_param_conf, train_conf)
                or satisfy_brake_condition(elem, i, vehicle_param_conf, train_conf)):
            elem[id_elem][0] = elem[i][0]
            elem[id_elem][1] = elem[i][1]
            elem[id_elem][2] = elem[i][2]
            elem[id_elem][3] = elem[i][3]  # add steering angle as reference
            id_elem += 1

    return elem[0:id_elem, 0:4]


def feature_distribute(elem, vehicle_param_conf, train_conf):
    """
    distribute feature into each grid
    """
    segment_cmd_list = gen_cmd_list(vehicle_param_conf, train_conf)
    segment_speed_list = np.linspace(
        train_conf.speed_min, train_conf.speed_max, num=train_conf.speed_segment).tolist()
    grid_dict = defaultdict(lambda: defaultdict(list))

    # stratified storing data
    feature_num = elem.shape[0]  # number of rows
    for feature_index in range(feature_num):
        cmd = elem[feature_index][2]  # cmd --- 2
        speed = elem[feature_index][0]  # speed --- 0
        for cmd_index in range(len(segment_cmd_list) - 1):
            curr_segment_cmd = segment_cmd_list[cmd_index]
            next_segment_cmd = segment_cmd_list[cmd_index + 1]
            if curr_segment_cmd < cmd < next_segment_cmd:
                for speed_index in range(len(segment_speed_list) - 1):
                    curr_segment_speed = segment_speed_list[speed_index]
                    next_segment_speed = segment_speed_list[speed_index + 1]
                    if curr_segment_speed < speed < next_segment_speed:
                        grid_dict[curr_segment_cmd][curr_segment_speed].append(feature_index)
                        break
                break

    # delete data which exceeds average value too much
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            feature_index_list = grid_dict[segment_cmd][segment_speed]
            if len(feature_index_list) == 0:
                continue
            acc_list = [elem[feature_index][1] for feature_index in feature_index_list]
            acc_mean = np.mean(acc_list)
            acc_std = np.std(acc_list)
            for index, feature_index in enumerate(feature_index_list):
                if abs(elem[feature_index][1] - acc_mean) > acc_std:
                    grid_dict[segment_cmd][segment_speed].pop(index)

    # random sampling data
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            feature_index_list = grid_dict[segment_cmd][segment_speed]
            store_num = min(len(feature_index_list), segment_store_num)
            feature_index_list = random.sample(feature_index_list, store_num)
            grid_dict[segment_cmd][segment_speed] = feature_index_list

    return (grid_dict, elem)


def feature_store(elem, vehicle_param_conf, train_conf):
    """
    store feature into segment_feature container
    """
    segment_cmd_list = gen_cmd_list(vehicle_param_conf, train_conf)
    segment_speed_list = np.linspace(
        train_conf.speed_min, train_conf.speed_max, num=train_conf.speed_segment).tolist()
    grid_dict = elem[0]
    feature = elem[1]
    segment_feature = np.zeros([len(feature), 3])
    counter = 0
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            for feature_index in grid_dict[segment_cmd][segment_speed]:
                # row: feature_index; col: 0:3
                segment_feature[counter] = feature[feature_index, 0:3]
                counter += 1
    return segment_feature[0:counter, :]


def gen_data(elems, vehicle_param_conf, train_conf):
    ret = feature_generate(elems, vehicle_param_conf)
    ret = calibration_table_utils.feature_filter(ret)
    ret = feature_cut(ret, vehicle_param_conf, train_conf)
    ret = feature_distribute(ret, vehicle_param_conf, train_conf)
    return feature_store(ret, vehicle_param_conf, train_conf)


def get_train_conf(folder_dir):
    train_conf_filename = 'calibration_table_conf.pb.txt'
    train_conf_file = os.path.join(folder_dir, train_conf_filename)
    train_conf = proto_utils.get_pb_from_text_file(train_conf_file, CalibrationTable())
    return train_conf
