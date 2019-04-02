#!/usr/bin/env python

import os
import sys

from keras.models import load_model
from scipy import interpolate
from scipy.signal import savgol_filter
import h5py
import numpy as np

from modules.common.configs.proto import vehicle_config_pb2
from fueling.control.dynamic_model.conf.model_config import feature_config, point_mass_config
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index, output_index
import common.proto_utils as proto_utils
import fueling.common.colored_glog as glog
import modules.control.proto.control_conf_pb2 as ControlConf

#Constants
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]
DIM_SEQUENCE_LENGTH = feature_config["sequence_length"]
DIM_DELAY_STEPS = feature_config["delay_steps"]
DELTA_T = feature_config["delta_t"]
MAXIMUM_SEGMENT_LENGTH = feature_config["maximum_segment_length"]
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]
CALIBRATION_DIMENSION = point_mass_config["calibration_dimension"]
VEHICLE_MODEL = point_mass_config["vehicle_model"]
STD_EPSILON = 1e-6


FILENAME_VEHICLE_PARAM_CONF = '/apollo/modules/common/data/vehicle_param.pb.txt'
VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(FILENAME_VEHICLE_PARAM_CONF,
                                            vehicle_config_pb2.VehicleConfig()).vehicle_param

FILENAME_CALIBRATION_TABLE_CONF = os.path.join(
    '/apollo/modules/calibration/data', VEHICLE_MODEL, 'control_conf.pb.txt')
CONTROL_CONF = proto_utils.get_pb_from_text_file(
    FILENAME_CALIBRATION_TABLE_CONF, ControlConf.ControlConf())

CALIBRATION_TABLE = CONTROL_CONF.lon_controller_conf.calibration_table
THROTTLE_DEADZONE = CONTROL_CONF.lon_controller_conf.throttle_deadzone
BRAKE_DEADZONE = CONTROL_CONF.lon_controller_conf.brake_deadzone


def generate_segment(h5_file):
    """
    load h5 file to a numpy array
    """
    segment = None
    glog.info('Loading {}'.format(h5_file))
    with h5py.File(h5_file, 'r') as fin:
        for ds in fin.itervalues():
            if segment is None:
                segment = np.array(ds)
            else:
                segment = np.concatenate((segment, np.array(ds)), axis=0)
    return segment


def feature_preprocessing(segment):
    """
    smooth noisy raw data from IMU by savgol_filter
    """
    # discard the segments that are too short
    if segment.shape[0] < WINDOW_SIZE or segment.shape[0] < DIM_DELAY_STEPS + DIM_SEQUENCE_LENGTH:
        return None
    # discard the segments that are too long
    elif segment.shape[0] > MAXIMUM_SEGMENT_LENGTH:
        return None
    else:
        # smooth IMU acceleration data
        segment[:, segment_index["a_x"]] = savgol_filter(
            segment[:, segment_index["a_x"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
        segment[:, segment_index["a_y"]] = savgol_filter(
            segment[:, segment_index["a_y"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
        return segment


def get_param_norm(input_feature, output_feature):
    """
    normalize the samples and save normalized parameters
    """
    glog.info("Input Feature Dimension {}".format(input_feature.shape))
    glog.info("Output Feature Dimension {}".format(output_feature.shape))
    glog.info("Start to calculate parameter norms")
    input_fea_mean = np.mean(input_feature, axis=0)
    input_fea_std = np.std(input_feature, axis=0) + STD_EPSILON
    output_fea_mean = np.mean(output_feature, axis=0)
    output_fea_std = np.std(output_feature, axis=0) + STD_EPSILON
    return (
        (input_fea_mean, input_fea_std),
        (output_fea_mean, output_fea_std)
    )


def generate_training_data(segment):
    """
    extract usable features from the numpy array for model training
    """
    total_len = segment.shape[0] - DIM_DELAY_STEPS
    total_sequence_num = segment.shape[0] - DIM_DELAY_STEPS - DIM_SEQUENCE_LENGTH
    glog.info('Total length: {}'.format(total_len))
    mlp_input_data = np.zeros([total_len, DIM_INPUT], order='C')
    mlp_output_data = np.zeros([total_len, DIM_OUTPUT], order='C')

    for k in range(segment.shape[0] - DIM_DELAY_STEPS):
        # speed mps
        mlp_input_data[k, input_index["speed"]] = segment[k, segment_index["speed"]]
        # acceleration
        mlp_input_data[k, input_index["acceleration"]] = \
            segment[k, segment_index["a_x"]] * np.cos(segment[k, segment_index["heading"]]) + \
            segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]])
        # throttle control from chassis
        mlp_input_data[k, input_index["throttle"]] = segment[k, segment_index["throttle"]]
        # brake control from chassis
        mlp_input_data[k, input_index["brake"]] = segment[k, segment_index["brake"]]
        # steering control from chassis
        mlp_input_data[k, input_index["steering"]] = segment[k, segment_index["steering"]]
        # acceleration next
        mlp_output_data[k, output_index["acceleration"]] = \
            segment[k + DIM_DELAY_STEPS, segment_index["a_x"]] * \
                np.cos(segment[k + DIM_DELAY_STEPS, segment_index["heading"]]) + \
            segment[k + DIM_DELAY_STEPS, segment_index["a_y"]] * \
                np.sin(segment[k + DIM_DELAY_STEPS, segment_index["heading"]])
        # angular speed next
        mlp_output_data[k, output_index["w_z"]] = \
            segment[k + DIM_DELAY_STEPS, segment_index["w_z"]]

    lstm_input_data = np.zeros([total_sequence_num, DIM_INPUT, DIM_SEQUENCE_LENGTH], order='C')
    lstm_output_data = np.zeros([total_sequence_num, DIM_OUTPUT], order='C')

    for k in range(mlp_input_data.shape[0] - DIM_SEQUENCE_LENGTH):
        lstm_input_data[k, :, :] = np.transpose(mlp_input_data[k:(k + DIM_SEQUENCE_LENGTH), :])
        lstm_output_data[k, :] = mlp_output_data[k + DIM_SEQUENCE_LENGTH, :]


    glog.info('mlp_input_data shape: {}'.format(mlp_input_data.shape))
    glog.info('mlp_output_data shape: {}'.format(mlp_output_data.shape))
    glog.info('lstm_input_data shape: {}'.format(lstm_input_data.shape))
    glog.info('lstm_output_data shape: {}'.format(lstm_output_data.shape))

    feature = [
        ("mlp_data", (mlp_input_data, mlp_output_data)),
        ("lstm_data", (lstm_input_data, lstm_output_data))
    ]
    return feature


def generate_evaluation_data(dataset_path, model_folder, model_name):
    segment = generate_segment(dataset_path)
    vehicle_state_gps, trajectory_gps = generate_gps_data(segment)
    output_imu = generate_imu_output(segment)
    output_point_mass = generate_point_mass_output(segment)
    output_fnn = generate_network_output(segment, model_folder, model_name)
    return vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps



def generate_gps_data(segment):
    # speed, heading by gps
    vehicle_state_gps = segment[:, [segment_index["speed"], segment_index["heading"]]]
    # position x, y by gps
    trajectory_gps = segment[:, [segment_index["x"], segment_index["y"]]] 
    return vehicle_state_gps, trajectory_gps


def generate_imu_output(segment):
    total_len = segment.shape[0]
    output_imu = np.zeros([total_len, DIM_OUTPUT])
    # acceleration by imu
    output_imu[:, output_index["acceleration"]] = \
            segment[:, segment_index["a_x"]] * np.cos(segment[:, segment_index["heading"]]) + \
            segment[:, segment_index["a_y"]] * np.sin(segment[:, segment_index["heading"]])
    output_imu[:, output_index["w_z"]] = segment[:, segment_index["w_z"]]  # angular speed by imu
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


def generate_point_mass_output(segment):
    calibration_table = load_calibration_table()
    table_interpolation = interpolate.interp2d(
        calibration_table[:, 0], calibration_table[:, 1], calibration_table[:, 2], kind='linear')
    
    total_len = segment.shape[0]
    velocity_point_mass = 0.0
    acceleration_point_mass = 0.0
    output_point_mass = np.zeros([total_len, DIM_OUTPUT])

    for k in range(total_len):
        if segment[k, segment_index["throttle"]] - THROTTLE_DEADZONE / 100.0 > \
                segment[k, segment_index["brake"]] - BRAKE_DEADZONE / 100.0: 
            lon_cmd = segment[k, segment_index["throttle"]]  # current cmd is throttle
        else:  
            lon_cmd = -segment[k, segment_index["brake"]]  # current cmd is brake

        if k == 0:
            velocity_point_mass = segment[k, segment_index["speed"]]
        else:
            velocity_point_mass += acceleration_point_mass * DELTA_T
        acceleration_point_mass = table_interpolation(velocity_point_mass, lon_cmd * 100.0)
        # acceleration by point_mass given by calibration table
        output_point_mass[k, output_index["acceleration"]] = acceleration_point_mass 
        # angular speed by point_mass given by linear bicycle model
        output_point_mass[k, output_index["w_z"]] = segment[k, segment_index["steering"]] * \
            VEHICLE_PARAM_CONF.max_steer_angle / VEHICLE_PARAM_CONF.steer_ratio * \
                velocity_point_mass / VEHICLE_PARAM_CONF.wheel_base
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
    acceleration_fnn = 0.0
    angular_velocity_fnn = 0.0

    for k in range(total_len):
        if k < DIM_SEQUENCE_LENGTH:
            velocity_fnn = segment[k, segment_index["speed"]]
            acceleration_fnn = segment[k, segment_index["a_x"]] * \
                np.cos(segment[k, segment_index["heading"]]) + \
                segment[k, segment_index["a_y"]] * np.sin(segment[k, segment_index["heading"]])
            angular_velocity_fnn = segment[k, segment_index["w_z"]]
            output_fnn[k, output_index["acceleration"]] = acceleration_fnn
            output_fnn[k, output_index["w_z"]] = angular_velocity_fnn

        if k >= DIM_SEQUENCE_LENGTH:
            if model_name == 'mlp':
                input_data_mlp[0, :] = input_data[k - 1, :]
                output_fnn[k, :] = model.predict(input_data_mlp)

            if model_name == 'lstm':
                input_data_array = np.reshape(np.transpose(
                        input_data[(k - DIM_SEQUENCE_LENGTH) : k, :]), 
                        (1, DIM_INPUT, DIM_SEQUENCE_LENGTH))
                output_fnn[k, :] = model.predict(input_data_array)
        
        output_fnn[k, :] = output_fnn[k, :] * output_std + output_mean

        acceleration_fnn = output_fnn[k, output_index["acceleration"]]
        angular_velocity_fnn = output_fnn[k, output_index["w_z"]]
        velocity_fnn += acceleration_fnn * DELTA_T

        input_data[k, input_index["speed"]] = velocity_fnn  # speed mps
        input_data[k, input_index["acceleration"]] = acceleration_fnn  # acceleration
        # throttle control from chassis
        input_data[k, input_index["throttle"]] = segment[k, segment_index["throttle"]]
        # brake control from chassis  
        input_data[k, input_index["brake"]] = segment[k, segment_index["brake"]]
        # steering control from chassis
        input_data[k, input_index["steering"]] = segment[k, segment_index["steering"]] 
        input_data[k, :] = (input_data[k, :] - input_mean) / input_std

    return output_fnn