#!/usr/bin/env python
# TODO: Fix order per README.md.
import glob
import math
import os
import random

import h5py
import numpy as np

from fueling.control.features.filters import Filters
from modules.common.configs.proto.vehicle_config_pb2 import VehicleParam
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.file_utils as file_utils
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


# # parameters
# WANTED_VEHICLE = 'Zhongyun'
# CONF_FOLDER = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
# vehicle_para_conf_filename = 'vehicle_param.pb.txt'
# conf_file = os.path.join(CONF_FOLDER, WANTED_VEHICLE, vehicle_para_conf_filename)
# VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(conf_file, VehicleParam())

conf_file_dir = '/apollo/modules/data/fuel/fueling/control/conf'
conf_filename = 'calibration_table_conf.pb.txt'
calibration_conf_file = os.path.join(conf_file_dir, conf_filename)
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(
    calibration_conf_file, CalibrationTable())
logging.info('Load calibration table conf: %s' % conf_filename)

# calibration table parameters
steer_condition = CALIBRATION_TABLE_CONF.steer_condition
curvature_condition = CALIBRATION_TABLE_CONF.curvature_condition

train_percentage = CALIBRATION_TABLE_CONF.train_percentage

THROTTLE_MAX = CALIBRATION_TABLE_CONF.throttle_max
BRAKE_MAX = -1 * CALIBRATION_TABLE_CONF.brake_max

speed_min_condition = CALIBRATION_TABLE_CONF.speed_min_condition
speed_max_condition = CALIBRATION_TABLE_CONF.speed_max_condition

segment_store_num = 12
wanted_driving_mode = "COMPLETE_MANUAL"

# vehicle parameters


def gen_brake_list(VEHICLE_PARAM_CONF):
    BRAKE_DEADZONE = -1 * VEHICLE_PARAM_CONF.brake_deadzone
    return np.linspace(
        BRAKE_MAX, BRAKE_DEADZONE, num=CALIBRATION_TABLE_CONF.brake_segment).tolist()


def gen_throttle_list(VEHICLE_PARAM_CONF):
    THROTTLE_DEADZONE = VEHICLE_PARAM_CONF.throttle_deadzone
    return np.linspace(
        THROTTLE_DEADZONE, THROTTLE_MAX, num=CALIBRATION_TABLE_CONF.throttle_segment).tolist()


def gen_cmd_list(VEHICLE_PARAM_CONF):
    segment_brake_list = gen_brake_list(VEHICLE_PARAM_CONF)
    segment_throttle_list = gen_throttle_list(VEHICLE_PARAM_CONF)
    return segment_brake_list + segment_throttle_list


segment_speed_list = np.linspace(
    CALIBRATION_TABLE_CONF.speed_min, CALIBRATION_TABLE_CONF.speed_max,
    num=CALIBRATION_TABLE_CONF.speed_segment).tolist()


def decide_cmd(chassis_throttle_val, chassis_brake_val, VEHICLE_PARAM_CONF):
    segment_throttle_list = gen_throttle_list(VEHICLE_PARAM_CONF)
    segment_brake_list = gen_brake_list(VEHICLE_PARAM_CONF)
    feature_cmd = 0.0
    if (VEHICLE_PARAM_CONF.vehicle_id.other_unique_id is "lexus"
            and chassis_throttle_val > 0.1 and chassis_brake_val > 0.1):
        logging.info("feature_cmd = 200")
        feature_cmd = 200.0
    if chassis_throttle_val > abs(segment_throttle_list[0]):
        feature_cmd = chassis_throttle_val
    elif chassis_brake_val > abs(segment_brake_list[-1]):
        feature_cmd = -chassis_brake_val
    else:
        feature_cmd = 0.0
    return feature_cmd


def feature_generate(elem, VEHICLE_PARAM_CONF):
    """ extract data from segment """
    res = np.zeros([len(elem), 5])
    for i in range(len(elem)):
        chassis = elem[i][0]
        pose = elem[i][1].pose

        heading_angle = pose.heading
        acc_x = pose.linear_acceleration.x
        acc_y = pose.linear_acceleration.y

        acc = acc_x * math.cos(heading_angle) + acc_y * math.sin(heading_angle)
        feature_cmd = decide_cmd(chassis.throttle_percentage,
                                 chassis.brake_percentage, VEHICLE_PARAM_CONF)
        driving_mode = (chassis.driving_mode == wanted_driving_mode)

        res[i] = np.array([
            chassis.speed_mps,  # 0: speed
            acc,  # 1: acc
            feature_cmd,                # 2: cmd
            chassis.steering_percentage,
            driving_mode
        ])
    return res


def feature_filter(elem, filter_window=20):
    """
    filter feature with mean filter
    feature is one colm of elem
    """
    # elem is a matrix:
    # row: different type of features,
    # col: different value of a feature
    num_row = elem.shape[0]  # data points in a feature
    for i in range(3):
        # feature_num: numbers of data in one feature
        feature = [0 for k in range(num_row)]
        f = Filters(filter_window)
        for j in range(num_row):
            value = elem[j][i]
            feature[j] = f.mean_filter(value)
        elem[:, i] = feature
    return elem


def satisfy_brake_condition(elem, index, VEHICLE_PARAM_CONF):
    """
    whether satisfy brake condition
    """
    acc_min_condition = VEHICLE_PARAM_CONF.max_deceleration
    segment_brake_list = gen_brake_list(VEHICLE_PARAM_CONF)
    condition = abs(elem[index][3]) < steer_condition and \
        elem[index][0] > speed_min_condition and \
        elem[index][0] < speed_max_condition and \
        elem[index][2] > segment_brake_list[0] and \
        elem[index][2] < segment_brake_list[-1] and \
        elem[index][1] < 0.0 and \
        elem[index][1] > acc_min_condition and \
        int(elem[index][4]) == 0
    return condition


def satisfy_throttle_condition(elem, index, VEHICLE_PARAM_CONF):
    """
    whether satisfy throttle condition
    """
    acc_max_condition = VEHICLE_PARAM_CONF.max_acceleration
    segment_throttle_list = gen_throttle_list(VEHICLE_PARAM_CONF)
    condition = abs(elem[index][3]) < steer_condition and \
        elem[index][0] > speed_min_condition and \
        elem[index][0] < speed_max_condition and \
        elem[index][2] > segment_throttle_list[0] and \
        elem[index][2] < segment_throttle_list[-1] and \
        elem[index][1] > 0.0 and \
        elem[index][1] < acc_max_condition and \
        int(elem[index][4]) == 0
    return condition


def feature_cut(elem, VEHICLE_PARAM_CONF):
    """
    get desired feature interval
    """
    id_elem = 0
    num_row = elem.shape[0]
    # find satisfied data
    for i in range(num_row):
        if satisfy_throttle_condition(elem, i, VEHICLE_PARAM_CONF) \
                or satisfy_brake_condition(elem, i, VEHICLE_PARAM_CONF):
            elem[id_elem][0] = elem[i][0]
            elem[id_elem][1] = elem[i][1]
            elem[id_elem][2] = elem[i][2]
            logging.info("elem_acc: %f" % elem[i][2])
            elem[id_elem][3] = elem[i][3]  # add steering angle as reference
            id_elem += 1

    return elem[0:id_elem, 0:4]


def feature_distribute(elem, VEHICLE_PARAM_CONF):
    """
    distribute feature into each grid
    """
    segment_cmd_list = gen_cmd_list(VEHICLE_PARAM_CONF)
    # TODO: Use collections.defaultdict(dict)
    grid_dict = {}
    for segment_cmd in segment_cmd_list:
        grid_dict[segment_cmd] = {}
        for segment_speed in segment_speed_list:
            grid_dict[segment_cmd][segment_speed] = []

    # stratified storing data
    feature_num = elem.shape[0]  # number of rows
    for feature_index in range(feature_num):
        cmd = elem[feature_index][2]  # cmd --- 2
        speed = elem[feature_index][0]  # speed --- 0
        for cmd_index in range(len(segment_cmd_list) - 1):
            curr_segment_cmd = segment_cmd_list[cmd_index]
            next_segment_cmd = segment_cmd_list[cmd_index + 1]
            if (cmd > curr_segment_cmd and cmd < next_segment_cmd):
                for speed_index in range(len(segment_speed_list) - 1):
                    curr_segment_speed = segment_speed_list[speed_index]
                    next_segment_speed = segment_speed_list[speed_index + 1]
                    if (speed > curr_segment_speed and speed < next_segment_speed):
                        # TODO: Allow 100 chars.
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


def feature_store(elem, VEHICLE_PARAM_CONF):
    """
    store feature into segment_feature container
    """
    segment_cmd_list = gen_cmd_list(VEHICLE_PARAM_CONF)
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


def write_h5_train_test(elem, origin_prefix, target_prefix):
    """write to h5 file"""
    (file_dir, key), features = elem
    feature_num = features.shape[0]  # row
    throttle_train_feature_num, throttle_test_feature_num = 0, 0
    brake_train_feature_num, brake_test_feature_num = 0, 0

    throttle_train = np.zeros(features.shape)
    throttle_test = np.zeros(features.shape)

    brake_train = np.zeros(features.shape)
    brake_test = np.zeros(features.shape)
    for feature in features:
        if feature[2] > 0.0:
            if random.random() < train_percentage:
                throttle_train[throttle_train_feature_num] = feature
                throttle_train_feature_num += 1
            else:
                throttle_test[throttle_test_feature_num] = feature
                throttle_test_feature_num += 1
        elif feature[2] < 0.0:
            if random.random() < train_percentage:
                brake_train[brake_train_feature_num] = feature
                brake_train_feature_num += 1
            else:
                brake_test[brake_test_feature_num] = feature
                brake_test_feature_num += 1

    # throttle train file
    logging.info('throttle file size: %d' % throttle_train.shape[0])

    # throttle train file
    throttle_train_target_prefix = os.path.join(target_prefix, 'throttle', 'train')
    throttle_train_file_dir = file_dir.replace(origin_prefix, throttle_train_target_prefix, 1)
    logging.info('Writing throttle_train hdf5 file to %s' % throttle_train_file_dir)
    throttle_train_data = throttle_train[0:throttle_train_feature_num, :]
    h5_utils.write_h5_single_segment(throttle_train_data, throttle_train_file_dir, key)

    # throttle test file
    throttle_test_target_prefix = os.path.join(target_prefix, 'throttle', 'test')
    throttle_test_file_dir = file_dir.replace(origin_prefix, throttle_test_target_prefix, 1)
    logging.info('Writing throttle_test hdf5 file to %s' % throttle_test_file_dir)
    throttle_test_data = throttle_test[0:throttle_test_feature_num, :]
    h5_utils.write_h5_single_segment(throttle_test_data, throttle_test_file_dir, key)

    # brake train file
    brake_train_target_prefix = os.path.join(target_prefix, 'brake', 'train')
    brake_train_file_dir = file_dir.replace(origin_prefix, brake_train_target_prefix, 1)
    logging.info('Writing brake_train hdf5 file to %s' % brake_train_file_dir)
    brake_train_data = brake_train[0:brake_train_feature_num, :]
    h5_utils.write_h5_single_segment(brake_train_data, brake_train_file_dir, key)

    # brake test file
    brake_test_target_prefix = os.path.join(target_prefix, 'brake', 'test')
    brake_test_file_dir = file_dir.replace(origin_prefix, brake_test_target_prefix, 1)
    logging.info('Writing brake_test hdf5 file to %s' % brake_test_file_dir)
    brake_test_data = brake_test[0:brake_test_feature_num, :]
    h5_utils.write_h5_single_segment(brake_test_data, brake_test_file_dir, key)

    return feature_num


def write_h5_cal_tab(data, file_dir, file_name):
    file_utils.makedirs(file_dir)
    file_name = file_name + '.hdf5'
    file_path = os.path.join(file_dir, file_name)
    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('segment', data=data, dtype='float64')
