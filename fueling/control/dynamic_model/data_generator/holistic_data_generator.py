#!/usr/bin/env python

import os

from keras.models import load_model
from scipy import interpolate
import h5py
import numpy as np

from fueling.control.dynamic_model.conf.model_config import imu_scaling
from fueling.control.dynamic_model.conf.model_config import feature_config, point_mass_config
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index, output_index
from fueling.control.dynamic_model.conf.model_config import holistic_input_index, holistic_output_index
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction

from modules.common.configs.proto import vehicle_config_pb2
import modules.control.proto.control_conf_pb2 as ControlConf

# Constants
PP6_IMU_SCALING = imu_scaling["pp6"]
PP7_IMU_SCALING = imu_scaling["pp7"]
DIM_INPUT = feature_config["holistic_input_dim"]
DIM_OUTPUT = feature_config["holistic_output_dim"]
DIM_SEQUENCE_LENGTH = feature_config["sequence_length"]
DIM_DELAY_STEPS = feature_config["delay_steps"]
DELTA_T = feature_config["delta_t"]
MAXIMUM_SEGMENT_LENGTH = feature_config["maximum_segment_length"]
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]
CALIBRATION_DIMENSION = point_mass_config["calibration_dimension"]
VEHICLE_MODEL = point_mass_config["vehicle_model"]
STD_EPSILON = 1e-6
SPEED_EPSILON = 1e-3   # Speed Threshold To Indicate Driving Directions

# TODO(ALL): Deprecate the hard-code gear status and read from data
GEAR_STATUS = 1  # 1: Driving Forward, 0: Natural, -1: Driving Backward

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
        # longitudinal speed mps
        mlp_input_data[k, holistic_input_index["lon_speed"]] = \
            segment[k, segment_index["v_x"]] * np.cos(segment[k, segment_index["heading"]]) + \
            segment[k, segment_index["v_y"]] * np.sin(segment[k, segment_index["heading"]])
        # lateral speed mps
        mlp_input_data[k, holistic_input_index["lat_speed"]] = \
            segment[k, segment_index["v_x"]] * np.sin(segment[k, segment_index["heading"]]) - \
            segment[k, segment_index["v_y"]] * np.cos(segment[k, segment_index["heading"]])
        # longitudinal acceleration
        mlp_input_data[k, holistic_input_index["lon_acceleration"]] = PP7_IMU_SCALING * \
            (segment[k, segment_index["a_x"]] * np.cos(segment[k, segment_index["heading"]]) +
             segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]]))
        # lateral acceleration
        mlp_input_data[k, holistic_input_index["lat_acceleration"]] = PP7_IMU_SCALING * \
            segment[k, segment_index["a_x"]] * np.sin(segment[k, segment_index["heading"]]) - \
            segment[k, segment_index["a_y"]] * np.cos(segment[k, segment_index["heading"]])
        # angular speed
        mlp_input_data[k, holistic_output_index["w_z"]] = PP7_IMU_SCALING * \
            segment[k, segment_index["w_z"]]

        # throttle control from chassis
        mlp_input_data[k, holistic_input_index["throttle"]] = segment[k, segment_index["throttle"]]
        # brake control from chassis
        mlp_input_data[k, holistic_input_index["brake"]] = segment[k, segment_index["brake"]]
        # steering control from chassis
        mlp_input_data[k, holistic_input_index["steering"]] = segment[k, segment_index["steering"]]

        # longitudinal acceleration next
        mlp_output_data[k, holistic_output_index["lon_acceleration"]] = PP7_IMU_SCALING * \
            (segment[k + DIM_DELAY_STEPS, segment_index["a_x"]] *
                np.cos(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]) +
             segment[k + DIM_DELAY_STEPS, segment_index["a_y"]] *
                np.sin(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]))
        # lateral acceleration next
        mlp_output_data[k, holistic_output_index["lat_acceleration"]] = PP7_IMU_SCALING * \
            (segment[k + DIM_DELAY_STEPS, segment_index["a_x"]] *
                np.sin(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]) -
             segment[k + DIM_DELAY_STEPS, segment_index["a_y"]] *
                np.cos(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]))
        # angular speed next
        mlp_output_data[k, holistic_output_index["w_z"]] = PP7_IMU_SCALING * \
            segment[k + DIM_DELAY_STEPS, segment_index["w_z"]]

    return mlp_input_data, mlp_output_data


def generate_gps_data(segment):
    total_len = segment.shape[0]
    vehicle_state_gps = np.zeros([total_len, DIM_OUTPUT])
    # longitudinal speed
    vehicle_state_gps[:, 0] = segment[:, segment_index["v_x"]] * \
        np.cos(segment[:, segment_index["heading"]]) + segment[:, segment_index["v_y"]] * \
        np.sin(segment[:, segment_index["heading"]])
    # lateral speed
    vehicle_state_gps[:, 1] = segment[:, segment_index["v_x"]] * \
        np.sin(segment[:, segment_index["heading"]]) - segment[:, segment_index["v_y"]] * \
        np.cos(segment[:, segment_index["heading"]])
    # vehicle heading
    vehicle_state_gps[:, 2] = segment[:, segment_index["heading"]]
    # position x, y by gps
    trajectory_gps = segment[:, [segment_index["x"], segment_index["y"]]]
    return vehicle_state_gps, trajectory_gps


def generate_imu_output(segment):
    total_len = segment.shape[0]
    output_imu = np.zeros([total_len, DIM_OUTPUT])
    # longitudinal acceleration by imu
    output_imu[:, holistic_output_index["lon_acceleration"]] = PP7_IMU_SCALING * \
        (segment[:, segment_index["a_x"]] * np.cos(segment[:, segment_index["heading"]]) +
         segment[:, segment_index["a_y"]] * np.sin(segment[:, segment_index["heading"]]))
    # lateral acceleration by imu
    output_imu[:, holistic_output_index["lat_acceleration"]] = PP7_IMU_SCALING * \
        (segment[:, segment_index["a_x"]] * np.sin(segment[:, segment_index["heading"]]) -
         segment[:, segment_index["a_y"]] * np.cos(segment[:, segment_index["heading"]]))
    output_imu[:, holistic_output_index["w_z"]] = segment[:, segment_index["w_z"]] * \
        PP7_IMU_SCALING  # angular speed by imu
    return output_imu


def load_calibration_table():
    table_length = len(CALIBRATION_TABLE.calibration)
    logging.info("Calibration Table Length: {}".format(table_length))
    calibration_table = np.zeros([table_length, CALIBRATION_DIMENSION])
    for i, calibration in enumerate(CALIBRATION_TABLE.calibration):
        calibration_table[i, 0] = calibration.speed
        calibration_table[i, 1] = calibration.command
        calibration_table[i, 2] = calibration.acceleration
    return calibration_table


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

        # If (negative speed under forward gear || positive speed under backward gear ||
        #     natural gear):
        # Then truncate speed, acceleration, and angular speed to 0
        if GEAR_STATUS * (velocity_point_mass + GEAR_STATUS * SPEED_EPSILON) <= 0:
            velocity_point_mass = 0.0
            output_point_mass[k, holistic_output_index["lon_acceleration"]] = 0.0
            output_point_mass[k, holistic_output_index["lat_acceleration"]] = 0.0
            output_point_mass[k, holistic_output_index["w_z"]] = 0.0
            continue

        # longitudinal acceleration by point_mass given by calibration table
        output_point_mass[k, holistic_output_index["lon_acceleration"]] = acceleration_point_mass
        # set lateral acceleration by point_mass to 0.0 by default
        output_point_mass[k, holistic_output_index["lat_acceleration"]] = 0.0
        # the angle between current velocity and vehicle heading
        velocity_angle_shift = np.arctan(REAR_WHEEL_BASE_PERCENTAGE * np.tan(
            segment[k, segment_index["steering"]] * MAX_STEER_ANGLE / STEER_RATIO))
        # angular speed by point_mass given by linear bicycle model
        output_point_mass[k, holistic_output_index["w_z"]] = velocity_point_mass * np.sin(
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

    lon_velocity_fnn = 0.0
    lat_velocity_fnn = 0.0

    for k in range(total_len):
        if k < DIM_SEQUENCE_LENGTH:
            lon_velocity_fnn = segment[k, segment_index["speed"]]
            # Scale the acceleration and angular speed data read from IMU
            output_fnn[k, holistic_output_index["lon_acceleration"]] = PP7_IMU_SCALING * (
                segment[k, segment_index["a_x"]] * np.cos(segment[k, segment_index["heading"]]) +
                segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]]))
            output_fnn[k, holistic_output_index["lat_acceleration"]] = PP7_IMU_SCALING * (
                segment[k, segment_index["a_x"]] * np.sin(segment[k, segment_index["heading"]]) -
                segment[k, segment_index["a_y"]] * np.cos(segment[k, segment_index["heading"]]))
            output_fnn[k, output_index["w_z"]] = PP7_IMU_SCALING * \
                segment[k, segment_index["w_z"]]

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
        lon_velocity_fnn += output_fnn[k, holistic_output_index["lon_acceleration"]] * DELTA_T
        lat_velocity_fnn += output_fnn[k, holistic_output_index["lat_acceleration"]] * DELTA_T
        # If (negative speed under forward gear || positive speed under backward gear ||
        #     natural gear):
        # Then truncate speed, acceleration, and angular speed to 0
        if GEAR_STATUS * (lon_velocity_fnn + GEAR_STATUS * SPEED_EPSILON) <= 0:
            lon_velocity_fnn = 0.0
            lat_velocity_fnn = 0.0
            output_fnn[k, holistic_output_index["lon_acceleration"]] = 0.0
            output_fnn[k, holistic_output_index["lat_acceleration"]] = 0.0
            output_fnn[k, holistic_output_index["w_z"]] = 0.0

        # longitudinal speed mps
        input_data[k, holistic_input_index["lon_speed"]] = lon_velocity_fnn
        # lateral speed mps
        input_data[k, holistic_input_index["lat_speed"]] = lat_velocity_fnn
        # longitudinal acceleration
        input_data[k, holistic_input_index["lon_acceleration"]
                   ] = output_fnn[k, holistic_output_index["lon_acceleration"]]
        # lateral acceleration
        input_data[k, holistic_input_index["lat_acceleration"]
                   ] = output_fnn[k, holistic_output_index["lat_acceleration"]]
        # angular velocity
        input_data[k, holistic_input_index["w_z"]] = output_fnn[k, holistic_output_index["w_z"]]
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
    vehicle_state_gps, trajectory_gps = generate_gps_data(segment)
    output_imu = generate_imu_output(segment)
    output_point_mass = generate_point_mass_output(segment)
    output_fnn = generate_network_output(segment, model_folder, model_name)
    return vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps
