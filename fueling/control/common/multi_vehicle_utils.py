#!/usr/bin/env python
""" utils for multiple vehicles """
import os
import time

import numpy as np

import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.control.proto.calibration_table_pb2 as CalibrationTable


FILENAME_CALIBRATION_TABLE_CONF = file_utils.fuel_path(
    'fueling/control/conf/calibration_table_conf.pb.txt')
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           CalibrationTable.CalibrationTable())

throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]

brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]

train_alpha = CALIBRATION_TABLE_CONF.train_alpha


def get_vehicle(path):
    return [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]


def get_vehicle_param(folder_dir):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(
        conf_file, vehicle_config_pb2.VehicleConfig())
    return VEHICLE_PARAM_CONF.vehicle_param


def gen_param_w_train_conf(vehicle_conf, train_conf, throttle_or_brake):
    if throttle_or_brake == 'throttle':
        cmd_min = vehicle_conf.throttle_deadzone
        cmd_max = train_conf.throttle_max
        layer = [train_conf.throttle_train_layer1,
                 train_conf.throttle_train_layer2,
                 train_conf.throttle_train_layer3]

    elif throttle_or_brake == 'brake':
        cmd_min = -1 * train_conf.brake_max
        cmd_max = -1 * vehicle_conf.brake_deadzone
        layer = [train_conf.brake_train_layer1,
                 train_conf.brake_train_layer2,
                 train_conf.brake_train_layer3]

    speed_min = train_conf.speed_min
    speed_max = train_conf.speed_max
    speed_segment_num = train_conf.train_speed_segment
    cmd_segment_num = train_conf.train_cmd_segment
    train_alpha = train_conf.train_alpha
    return ((speed_min, speed_max, speed_segment_num),
            (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha)


def gen_param(vehicle_param, throttle_or_brake):
    if throttle_or_brake == 'throttle':
        cmd_min = vehicle_param.throttle_deadzone
        cmd_max = CALIBRATION_TABLE_CONF.throttle_max
        layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                 CALIBRATION_TABLE_CONF.throttle_train_layer2,
                 CALIBRATION_TABLE_CONF.throttle_train_layer3]

    elif throttle_or_brake == 'brake':
        cmd_min = -1 * CALIBRATION_TABLE_CONF.brake_max
        cmd_max = -1 * vehicle_param.brake_deadzone
        layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                 CALIBRATION_TABLE_CONF.brake_train_layer2,
                 CALIBRATION_TABLE_CONF.brake_train_layer3]

    speed_min = CALIBRATION_TABLE_CONF.train_speed_min
    speed_max = CALIBRATION_TABLE_CONF.train_speed_max
    speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment
    cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment
    train_alpha = CALIBRATION_TABLE_CONF.train_alpha
    return ((speed_min, speed_max, speed_segment_num),
            (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha)
