#!/usr/bin/env python

import os
import sys

from keras.models import load_model
from scipy import interpolate
from scipy.signal import savgol_filter
import colored_glog as glog
import h5py
import numpy as np


from fueling.control.dynamic_model.conf.model_config import acc_method, imu_scaling, pose_output_index
from fueling.control.dynamic_model.conf.model_config import feature_config, point_mass_config
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index, output_index
from fueling.control.dynamic_model.conf.model_config import holistic_input_index, holistic_output_index
import fueling.common.proto_utils as proto_utils
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.utils.echo_lincoln as echo_lincoln

from modules.common.configs.proto import vehicle_config_pb2
import modules.control.proto.control_conf_pb2 as ControlConf


# Constants
if acc_method["acc_from_IMU"]:
    PP7_IMU_SCALING = imu_scaling["pp7"]
else:
    PP7_IMU_SCALING = 1.0

PP6_IMU_SCALING = imu_scaling["pp6"]
IS_HOLISTIC = feature_config["is_holistic"]
DIM_INPUT = feature_config["holistic_input_dim"] if IS_HOLISTIC else feature_config["input_dim"]
DIM_OUTPUT = feature_config["holistic_output_dim"] if IS_HOLISTIC else feature_config["output_dim"]
DIM_SEQUENCE_LENGTH = feature_config["sequence_length"]
DIM_DELAY_STEPS = feature_config["delay_steps"]
DELTA_T = feature_config["delta_t"]
MAXIMUM_SEGMENT_LENGTH = feature_config["maximum_segment_length"]
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]
CALIBRATION_DIMENSION = point_mass_config["calibration_dimension"]
VEHICLE_MODEL = point_mass_config["vehicle_model"]
STD_EPSILON = 1e-6
SPEED_EPSILON = 1e-6   # Speed Threshold To Indicate Driving Directions

FILENAME_VEHICLE_PARAM_CONF = '/apollo/modules/common/data/vehicle_param.pb.txt'
VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(FILENAME_VEHICLE_PARAM_CONF,
                                                       vehicle_config_pb2.VehicleConfig())
FILENAME_CALIBRATION_TABLE_CONF = os.path.join(
    '/apollo/modules/calibration/data', VEHICLE_MODEL, 'control_conf.pb.txt')
CONTROL_CONF = proto_utils.get_pb_from_text_file(
    FILENAME_CALIBRATION_TABLE_CONF, ControlConf.ControlConf())

CALIBRATION_TABLE = CONTROL_CONF.lon_controller_conf.calibration_table
THROTTLE_DEADZONE = VEHICLE_PARAM_CONF.vehicle_param.throttle_deadzone
BRAKE_DEADZONE = VEHICLE_PARAM_CONF.vehicle_param.brake_deadzone
MAX_STEER_ANGLE = VEHICLE_PARAM_CONF.vehicle_param.max_steer_angle
STEER_RATIO = VEHICLE_PARAM_CONF.vehicle_param.steer_ratio
WHEEL_BASE = VEHICLE_PARAM_CONF.vehicle_param.wheel_base
REAR_WHEEL_BASE_PERCENTAGE = 0.3


def generate_mlp_data(segment, total_len):
    mlp_input_data = np.zeros([total_len, DIM_INPUT], order='C')
    mlp_output_data = np.zeros([total_len, DIM_OUTPUT], order='C')
    for k in range(segment.shape[0] - DIM_DELAY_STEPS):
        # speed mps
        # mlp_input_data[k, input_index["speed"]] = segment[k, segment_index["speed"]]
        mlp_input_data[k, input_index["speed"]] = (
            segment[k, segment_index["v_x"]] * np.cos(segment[k, segment_index["heading"]])
            + segment[k, segment_index["v_y"]] * np.sin(segment[k, segment_index["heading"]]))
        # acceleration
        mlp_input_data[k, input_index["acceleration"]] = PP7_IMU_SCALING * (
            segment[k, segment_index["a_x"]] * np.cos(segment[k, segment_index["heading"]])
            + segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]]))
        # throttle control from chassis
        mlp_input_data[k, input_index["throttle"]] = segment[k, segment_index["throttle"]]
        # brake control from chassis
        mlp_input_data[k, input_index["brake"]] = segment[k, segment_index["brake"]]
        # steering control from chassis
        mlp_input_data[k, input_index["steering"]] = segment[k, segment_index["steering"]]
        # acceleration next
        mlp_output_data[k, output_index["acceleration"]] = PP7_IMU_SCALING * (
            segment[k + DIM_DELAY_STEPS, segment_index["a_x"]]
            * np.cos(segment[k + DIM_DELAY_STEPS, segment_index["heading"]])
            + segment[k + DIM_DELAY_STEPS, segment_index["a_y"]]
            * np.sin(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]))
        # angular speed next
        mlp_output_data[k, output_index["w_z"]] = PP7_IMU_SCALING * \
            segment[k + DIM_DELAY_STEPS, segment_index["w_z"]]
    return mlp_input_data, mlp_output_data


def generate_gps_data(segment):
    total_len = segment.shape[0]
    vehicle_state_gps = np.zeros([total_len, DIM_OUTPUT])
    # speed, heading by gps
    vehicle_state_gps[:, 0] = (
        segment[:, segment_index["v_x"]] * np.cos(segment[:, segment_index["heading"]])
        + segment[:, segment_index["v_y"]] * np.sin(segment[:, segment_index["heading"]]))
    vehicle_state_gps[:, 1] = segment[:, segment_index["heading"]]
    # position x, y by gps
    trajectory_gps = segment[:, [segment_index["x"], segment_index["y"]]]
    return vehicle_state_gps, trajectory_gps


def generate_imu_output(segment):
    total_len = segment.shape[0]
    output_imu = np.zeros([total_len, DIM_OUTPUT])
    # acceleration by imu
    output_imu[:, output_index["acceleration"]] = PP7_IMU_SCALING * \
        (segment[:, segment_index["a_x"]] * np.cos(segment[:, segment_index["heading"]])
         + segment[:, segment_index["a_y"]] * np.sin(segment[:, segment_index["heading"]]))
    # angular speed by imu
    output_imu[:, output_index["w_z"]] = segment[:, segment_index["w_z"]] * PP7_IMU_SCALING
    return output_imu


def generate_imu_output_wo_PP7(segment):
    total_len = segment.shape[0]
    output_imu = np.zeros([total_len, DIM_OUTPUT])
    # acceleration by imu
    output_imu[:, output_index["acceleration"]] = \
        (segment[:, segment_index["a_x"]] * np.cos(segment[:, segment_index["heading"]])
         + segment[:, segment_index["a_y"]] * np.sin(segment[:, segment_index["heading"]]))
    # angular speed by imu
    output_imu[:, output_index["w_z"]] = segment[:, segment_index["w_z"]]
    return output_imu


def load_calibration_table():
    table_length = len(CALIBRATION_TABLE.calibration)
    glog.info("Calibration Table Length: {}".format(table_length))
    calibration_table = np.zeros([table_length, CALIBRATION_DIMENSION])
    for i, calibration in enumerate(CALIBRATION_TABLE.calibration):
        calibration_table[i, 0] = calibration.speed
        calibration_table[i, 1] = calibration.command
        calibration_table[i, 2] = calibration.acceleration
    return calibration_table


def gear_position_conversion(raw_gear):
    gear_status = np.round(raw_gear)
    # Convert backward gear from 2 to -1 for computation convenience
    if gear_status == 2:
        gear_status = -1
    # Set forward gear as 1
    elif gear_status == 1:
        pass
    # Set neutral gear as 0
    else:
        gear_status = 0
    return gear_status


def generate_point_mass_output(segment):
    calibration_table = load_calibration_table()
    table_interpolation = interpolate.interp2d(
        calibration_table[:, 0], calibration_table[:, 1], calibration_table[:, 2], kind='linear')

    total_len = segment.shape[0]
    velocity_point_mass = 0.0
    velocity_angle_shift = 0.0
    acceleration_point_mass = 0.0
    output_point_mass = np.zeros([total_len, DIM_OUTPUT])

    for k in range(total_len):
        if segment[k, segment_index["throttle"]] - THROTTLE_DEADZONE / 100.0 > \
                segment[k, segment_index["brake"]] - BRAKE_DEADZONE / 100.0:
            # current cmd is throttle
            lon_cmd = segment[k, segment_index["throttle"]]
        else:
            # current cmd is brake
            lon_cmd = -segment[k, segment_index["brake"]]

        if k == 0:
            velocity_point_mass = segment[k, segment_index["speed"]]

        acceleration_point_mass = table_interpolation(velocity_point_mass, lon_cmd * 100.0)
        velocity_point_mass += acceleration_point_mass * DELTA_T

        # Get gear status from data, default status is forward driving gear
        # 0: Neutral, 1: Driving Forward, 2: Driving Backward
        if segment.shape[1] > segment_index["gear_position"]:
            gear_status = gear_position_conversion(segment[k, segment_index["gear_position"]])
        else:
            gear_status = 1
        # If (negative speed under forward gear || positive speed under backward gear ||
        #     natural gear):
        # Then truncate speed, acceleration, and angular speed to 0
        if gear_status * (velocity_point_mass + gear_status * SPEED_EPSILON) <= 0:
            velocity_point_mass = 0.0
            output_point_mass[k, output_index["acceleration"]] = 0.0
            output_point_mass[k, output_index["w_z"]] = 0.0
            continue

        # acceleration by point_mass given by calibration table
        output_point_mass[k, output_index["acceleration"]] = acceleration_point_mass
        # the angle between current velocity and vehicle heading
        velocity_angle_shift = np.arctan(REAR_WHEEL_BASE_PERCENTAGE * np.tan(
            segment[k, segment_index["steering"]] * MAX_STEER_ANGLE / STEER_RATIO))
        # angular speed by point_mass given by linear bicycle model
        output_point_mass[k, output_index["w_z"]] = velocity_point_mass * np.sin(
            velocity_angle_shift) / (WHEEL_BASE * REAR_WHEEL_BASE_PERCENTAGE)
    return output_point_mass


def generate_network_output(segment, model_folder, model_name):
    model_norms_path = os.path.join(model_folder, 'norms.h5')
    with h5py.File(model_norms_path, 'r') as model_norms_file:
        input_mean = np.array(model_norms_file.get('input_mean'))
        input_std = np.array(model_norms_file.get('input_std'))
        output_mean = np.array(model_norms_file.get('output_mean'))
        output_std = np.array(model_norms_file.get('output_std'))

    model_weights_path = os.path.join(model_folder, 'weights.h5')
    model = load_model(model_weights_path)

    total_len = segment.shape[0]
    input_data = np.zeros([total_len, DIM_INPUT])
    input_data_mlp = np.zeros([1, DIM_INPUT])
    output_fnn = np.zeros([total_len, DIM_OUTPUT])

    velocity_fnn = 0.0

    for k in range(total_len):
        if k < DIM_SEQUENCE_LENGTH:
            velocity_fnn = segment[k, segment_index["speed"]]
            # Scale the acceleration and angular speed data read from IMU
            output_fnn[k, output_index["acceleration"]] = PP7_IMU_SCALING * (
                segment[k, segment_index["a_x"]] * np.cos(segment[k, segment_index["heading"]]) +
                segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]]))
            output_fnn[k, output_index["w_z"]] = PP7_IMU_SCALING * segment[k, segment_index["w_z"]]

        if k >= DIM_SEQUENCE_LENGTH:
            if model_name == 'mlp':
                input_data_mlp[0, :] = input_data[k - 1, :]
                output_fnn[k, :] = model.predict(input_data_mlp)

            if model_name == 'lstm':
                input_data_array = np.reshape(np.transpose(
                    input_data[(k - DIM_SEQUENCE_LENGTH): k, :]),
                    (1, DIM_INPUT, DIM_SEQUENCE_LENGTH))
                output_fnn[k, :] = model.predict(input_data_array)

        output_fnn[k, :] = output_fnn[k, :] * output_std + output_mean

        # Update the vehicle speed based on predicted acceleration
        velocity_fnn += output_fnn[k, output_index["acceleration"]] * DELTA_T

        # Get raw gear status from data, default status is forward driving gear
        # Before conversion: 1: Driving Forward, 2: Driving Backward, 3: Neutral
        # After conversion: 1: Driving Forward, -1: Driving Backward,  0: Neutral
        if segment.shape[1] > segment_index["gear_position"]:
            gear_status = gear_position_conversion(segment[k, segment_index["gear_position"]])
        else:
            gear_status = 1
        # If (negative speed under forward gear || positive speed under backward gear ||
        #     neutral gear):
        # Then truncate speed, acceleration, and angular speed to 0
        if gear_status * (velocity_fnn + gear_status * SPEED_EPSILON) <= 0:
            velocity_fnn = 0.0
            output_fnn[k, output_index["acceleration"]] = 0.0
            output_fnn[k, output_index["w_z"]] = 0.0

        input_data[k, input_index["speed"]] = velocity_fnn  # speed mps
        # acceleration
        input_data[k, input_index["acceleration"]] = output_fnn[k, output_index["acceleration"]]
        # throttle control from chassis
        input_data[k, input_index["throttle"]] = segment[k, segment_index["throttle"]]
        # brake control from chassis
        input_data[k, input_index["brake"]] = segment[k, segment_index["brake"]]
        # steering control from chassis
        input_data[k, input_index["steering"]] = segment[k, segment_index["steering"]]
        input_data[k, :] = (input_data[k, :] - input_mean) / input_std

    return output_fnn


def generate_evaluation_data(dataset_path, model_folder, model_name):
    segment = feature_extraction.generate_segment(dataset_path)
    segment = feature_extraction.feature_preprocessing(segment)
    if not segment.any():
        glog.error('Errors occur during evaluation data generation')
        sys.exit()
    vehicle_state_gps, trajectory_gps = generate_gps_data(segment)
    output_echo_lincoln = echo_lincoln.echo_lincoln_wrapper(dataset_path)
    output_imu = generate_imu_output(segment)
    output_point_mass = generate_point_mass_output(segment)
    output_fnn = generate_network_output(segment, model_folder, model_name)
    return vehicle_state_gps, output_echo_lincoln, output_imu, output_point_mass, \
        output_fnn, trajectory_gps
