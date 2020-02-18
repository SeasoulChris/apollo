#!/usr/bin/env python

from math import sqrt
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

from google.protobuf import text_format
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


from fueling.control.dynamic_model.conf.model_config import acc_method, imu_scaling
from fueling.control.dynamic_model.conf.model_config import feature_config
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index, output_index
from fueling.control.proto.dynamic_model_evaluation_pb2 import EvaluationResults
import fueling.common.logging as logging
import fueling.control.dynamic_model.data_generator.non_holistic_data_generator as data_generator


# System setup
USE_TENSORFLOW = True  # Slightly faster than Theano.
USE_GPU = False  # CPU seems to be faster than GPU in this case.
ALPHA = 0.7  # transparent coef of plot

if USE_TENSORFLOW:
    if not USE_GPU:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["KERAS_BACKEND"] = "tensorflow"
    from keras.callbacks import TensorBoard
else:
    os.environ["KERAS_BACKEND"] = "theano"
    if USE_GPU:
        os.environ["THEANORC"] = os.path.join(
            os.getcwd(), "theanorc/gpu_config")
        os.environ["DEVICE"] = "cuda"  # for pygpu, unclear whether necessary
    else:
        os.environ["THEANORC"] = os.path.join(
            os.getcwd(), "theanorc/cpu_config")

# Constants
IS_BACKWARD = feature_config["is_backward"]
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]
DIM_LSTM_LENGTH = feature_config["sequence_length"]
DELTA_T = feature_config["delta_t"]


def heading_angle(scenario_segments, platform_path):
    """ plot heading angles """
    (scenario, (segment_origin, segment_d, segment_dd)) = scenario_segments
    alpha_origin = segment_origin[:, segment_index["heading"]]
    alpha_integral = np.zeros(alpha_origin.shape)
    segment_d[0, segment_index["heading"]] = segment_origin[0, segment_index["heading"]]
    segment_dd[0, segment_index["heading"]] = segment_origin[0, segment_index["heading"]]
    for index in range(1, segment_d.shape[0]):
        segment_d[index, segment_index["heading"]] = normalize_angle(
            segment_d[index - 1, segment_index["heading"]] +
            segment_d[index - 1, segment_index["w_z"]] * DELTA_T)
        segment_dd[index, segment_index["heading"]] = normalize_angle(
            segment_dd[index - 1, segment_index["heading"]] +
            segment_dd[index - 1, segment_index["w_z"]] * DELTA_T)
    # plot
    pdf_file_path = os.path.join(platform_path, "heading_angle{}.pdf".format(scenario))
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.plot(alpha_origin, "b--", label="Origin Heading Angle", linewidth=1)
        plt.plot(segment_d[:, segment_index["heading"]], color="red",
                 label="Integrated Heading Angle", linewidth=0.5)
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()
    return pdf_file_path


def location(scenario_segments, platform_path):
    """ plot location """
    speed(scenario_segments, platform_path)
    (scenario, (segment_origin, segment_d, segment_dd)) = scenario_segments
    location_x = segment_origin[:, segment_index["x"]]
    location_y = segment_origin[:, segment_index["y"]]
    segment_d[0, segment_index["x"]] = segment_origin[0, segment_index["x"]]
    segment_d[0, segment_index["y"]] = segment_origin[0, segment_index["y"]]
    segment_dd[0, segment_index["x"]] = segment_origin[0, segment_index["x"]]
    segment_dd[0, segment_index["y"]] = segment_origin[0, segment_index["y"]]
    # integration from IMU
    tmp_x = np.zeros(location_x.shape)
    tmp_y = np.zeros(location_y.shape)
    tmp_x[0] = segment_origin[0, segment_index["x"]]
    tmp_y[0] = segment_origin[0, segment_index["y"]]
    for index in range(1, len(location_x)):
        # location from speed (acc from speed)
        segment_d[index, segment_index["x"]] = (
            segment_d[index - 1, segment_index["x"]] +
            segment_d[index - 1, segment_index["v_x"]] * DELTA_T)
        segment_d[index, segment_index["y"]] = (
            segment_d[index - 1, segment_index["y"]] +
            segment_d[index - 1, segment_index["v_y"]] * DELTA_T)
        # location from acc(speed)
        segment_dd[index, segment_index["x"]] = \
            segment_dd[index - 1, segment_index["x"]] + \
            segment_dd[index - 1, segment_index["v_x"]] * DELTA_T
        segment_dd[index, segment_index["y"]] = \
            segment_dd[index - 1, segment_index["y"]] + \
            segment_dd[index - 1, segment_index["v_y"]] * DELTA_T
        # location from origin IMU
        tmp_v_x = segment_origin[index - 1, segment_index["v_x"]]
        tmp_v_y = segment_origin[index - 1, segment_index["v_y"]]
        logging.info("tmp_v_x: {}".format(tmp_v_x))
        tmp_x[index] = tmp_x[index - 1] + tmp_v_x * DELTA_T +\
            1 / 2 * imu_scaling["pp7"] * segment_origin[index -
                                                        1, segment_index["a_x"]] * DELTA_T * DELTA_T
        tmp_y[index] = tmp_y[index - 1] + tmp_v_y * DELTA_T +\
            1 / 2 * imu_scaling["pp7"] * segment_origin[index -
                                                        1, segment_index["a_y"]] * DELTA_T * DELTA_T
    pdf_file_path = os.path.join(platform_path, "location{}.pdf".format(scenario))
    logging.info("tmp_x shape {}".format(tmp_x.shape))
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.plot(segment_origin[:, segment_index["x"]],
                 segment_origin[:, segment_index["y"]],
                 "y--",
                 alpha=ALPHA,
                 label="Direct location",
                 linewidth=1)
        plt.plot(segment_d[:, segment_index["x"]],
                 segment_d[:, segment_index["y"]],
                 color="blue",
                 alpha=ALPHA,
                 label="location_1",
                 linewidth=1)
        plt.plot(segment_dd[:, segment_index["x"]],
                 segment_dd[:, segment_index["y"]],
                 "r--",
                 label="location_2",
                 linewidth=0.3)
        plt.plot(tmp_x[:], tmp_y[:],
                 "g--",
                 label="location_3",
                 linewidth=0.4)
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

    return pdf_file_path


def speed(scenario_segments, platform_path):
    """ plot speed """
    acceleration(scenario_segments, platform_path)
    (scenario, (segment_origin, segment_d, segment_dd)) = scenario_segments
    v_origin = np.zeros(segment_origin[:, segment_index["v_x"]].shape)
    v_d = np.zeros(segment_origin[:, segment_index["v_x"]].shape)
    v_dd = np.zeros(segment_origin[:, segment_index["v_x"]].shape)
    for index in range(0, len(v_origin) - 1):
        # origin speed
        v_origin[index] = (segment_origin[index, segment_index["v_x"]] *
                           np.cos(normalize_angle(segment_origin[index, segment_index["heading"]])) +
                           segment_origin[index, segment_index["v_y"]] *
                           np.sin(normalize_angle(segment_origin[index, segment_index["heading"]])))
        # differential from location
        if index == 0:
            segment_d[index, segment_index["v_x"]] = segment_origin[index, segment_index["v_x"]]
            segment_dd[index, segment_index["v_x"]] = segment_origin[index, segment_index["v_x"]]
            segment_d[index, segment_index["v_y"]] = segment_origin[index, segment_index["v_y"]]
            segment_dd[index, segment_index["v_y"]] = segment_origin[index, segment_index["v_y"]]
            v_d[index] = (segment_d[index, segment_index["v_x"]] *
                          np.cos(normalize_angle(segment_d[index, segment_index["heading"]])) +
                          segment_d[index, segment_index["v_y"]] *
                          np.sin(normalize_angle(segment_d[index, segment_index["heading"]])))
            v_dd[index] = (segment_dd[index, segment_index["v_x"]] *
                           np.cos(normalize_angle(segment_dd[index, segment_index["heading"]])) +
                           segment_dd[index, segment_index["v_y"]] *
                           np.sin(normalize_angle(segment_dd[index, segment_index["heading"]])))
        else:
            segment_dd[index, segment_index["v_x"]] = (
                segment_dd[index - 1, segment_index["v_x"]] +
                segment_dd[index - 1, segment_index["a_x"]] * DELTA_T)
            segment_dd[index, segment_index["v_y"]] = (
                segment_dd[index - 1, segment_index["v_y"]] +
                segment_dd[index - 1, segment_index["a_y"]] * DELTA_T)

            a_d = (segment_d[index - 1, segment_index["a_x"]] *
                   np.cos(normalize_angle(segment_d[index - 1, segment_index["heading"]])) +
                   segment_d[index - 1, segment_index["a_y"]] *
                   np.sin(normalize_angle(segment_d[index - 1, segment_index["heading"]])))
            v_d[index] = v_d[index - 1] + a_d * DELTA_T

            a_dd = (segment_dd[index - 1, segment_index["a_x"]] *
                    np.cos(normalize_angle(segment_dd[index, segment_index["heading"]])) +
                    segment_dd[index - 1, segment_index["a_y"]] *
                    np.sin(normalize_angle(segment_dd[index, segment_index["heading"]])))
            v_dd[index] = v_dd[index - 1] + a_dd * DELTA_T

    # plot
    pdf_file_path = os.path.join(platform_path, "speed{}.pdf".format(scenario))
    logging.info("max: {}".format(max(abs(v_origin - v_d))))
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.plot(v_origin, "y--", alpha=ALPHA, label="Direct Speed", linewidth=1)
        plt.plot(v_d, color="blue", label="Speed_1", linewidth=0.5)
        plt.plot(v_dd, "r--", alpha=ALPHA, label="Speed_2", linewidth=0.3)
        plt.legend()
        legend = plt.legend(loc='upper right', fontsize='small')
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

    return pdf_file_path


def acceleration(scenario_segments, platform_path):
    """ plot acc """
    heading_angle(scenario_segments, platform_path)
    (scenario, (segment_origin, segment_d, segment_dd)) = scenario_segments
    acc_origin = np.zeros(segment_origin[:, segment_index["a_x"]].shape)
    acc_d = np.zeros(segment_origin[:, segment_index["a_x"]].shape)
    acc_dd = np.zeros(segment_origin[:, segment_index["a_x"]].shape)
    logging.info("max: {}".format(
        max(abs(segment_origin[:, segment_index["a_x"]] - segment_d[:, segment_index["a_x"]]))))
    # origin
    for index in range(0, len(acc_origin)):
        acc_origin[index] = (segment_origin[index, segment_index["a_x"]] *
                             np.cos(normalize_angle(segment_origin[index, segment_index["heading"]])) +
                             segment_origin[index, segment_index["a_y"]] *
                             np.sin(normalize_angle(segment_origin[index, segment_index["heading"]])))
        # acc from speed
        acc_d[index] = (segment_d[index, segment_index["a_x"]] *
                        np.cos(normalize_angle(segment_d[index, segment_index["heading"]])) +
                        segment_d[index, segment_index["a_y"]] *
                        np.sin(normalize_angle(segment_d[index, segment_index["heading"]])))
        # acc from location
        acc_dd[index] = (segment_dd[index, segment_index["a_x"]] *
                         np.cos(normalize_angle(segment_dd[index, segment_index["heading"]])) +
                         segment_dd[index, segment_index["a_y"]] *
                         np.sin(normalize_angle(segment_dd[index, segment_index["heading"]])))
    # plot
    pdf_file_path = os.path.join(platform_path, "acc{}.pdf".format(scenario))
    logging.info("max: {}".format(max(abs(acc_origin - acc_d))))
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.plot(acc_origin, "y--", alpha=ALPHA * 0.5, label="Direct Acceleration", linewidth=1)
        plt.plot(acc_d, color="blue", label="Acc_1", linewidth=0.5)
        plt.plot(acc_dd, "r--", alpha=ALPHA, label="Acc_2", linewidth=0.3)
        plt.legend()
        legend = plt.legend(loc='upper right', fontsize='small')
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

    return pdf_file_path


def evaluate_direct_output(output_imu, output_fnn, output_point_mass, evaluation_results):

    rmse_fnn_acceleration = sqrt(mean_squared_error(output_imu[:, 0], output_fnn[:, 0]))
    rmse_point_mass_acceleration = sqrt(mean_squared_error(
        output_imu[:, 0], output_point_mass[:, 0]))
    rms_acceleration = sqrt(sum(n * n for n in output_imu[:, 0]) / len(output_imu[:, 0]))
    evaluation_results.learning_based_result.acceleration_error = rmse_fnn_acceleration
    evaluation_results.learning_based_result.acceleration_error_rate = \
        rmse_fnn_acceleration / rms_acceleration
    evaluation_results.point_mass_result.acceleration_error = rmse_point_mass_acceleration
    evaluation_results.point_mass_result.acceleration_error_rate = \
        rmse_point_mass_acceleration / rms_acceleration

    rmse_fnn_angular_speed = sqrt(mean_squared_error(output_imu[:, 1], output_fnn[:, 1]))
    rmse_point_mass_angular_speed = sqrt(
        mean_squared_error(output_imu[:, 1], output_point_mass[:, 1]))
    rms_angular_speed = sqrt(sum(n * n for n in output_imu[:, 1]) / len(output_imu[:, 1]))
    evaluation_results.learning_based_result.angular_speed_error = rmse_fnn_angular_speed
    evaluation_results.learning_based_result.angular_speed_error_rate = \
        rmse_fnn_angular_speed / rms_angular_speed
    evaluation_results.point_mass_result.angular_speed_error = rmse_point_mass_angular_speed
    evaluation_results.point_mass_result.angular_speed_error_rate = \
        rmse_point_mass_angular_speed / rms_angular_speed


def normalize_angle(theta):
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta = theta - 2 * math.pi
    return theta

# echo_lincoln output: acc, angular_velocity, speed


def evaluate_vehicle_state(vehicle_state_gps, output_echo_lincoln, output_imu, output_fnn,
                           output_point_mass, evaluation_results):
    vehicle_state_echo_lincoln = np.zeros(
        [vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])

    vehicle_state_imu = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_fnn = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_point_mass = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_echo_lincoln[0, 0:2] = vehicle_state_gps[0, :]

    vehicle_state_imu[0, :] = vehicle_state_gps[0, :]
    vehicle_state_fnn[0, :] = vehicle_state_gps[0, :]
    vehicle_state_point_mass[0, :] = vehicle_state_gps[0, :]

    for index in range(1, vehicle_state_gps.shape[0]):
        # vehicle states by echo_lincoln
        vehicle_state_echo_lincoln[index, 0:2] = vehicle_state_echo_lincoln[index - 1, 0:2] + \
            output_echo_lincoln[index - 1, 0:2] * DELTA_T
        vehicle_state_echo_lincoln[index, 1] = normalize_angle(vehicle_state_echo_lincoln[index, 1])
        # vehicle states by imu sensor
        vehicle_state_imu[index, :] = vehicle_state_imu[index - 1, :] + \
            output_imu[index, :] * DELTA_T
        vehicle_state_imu[index, 1] = normalize_angle(vehicle_state_imu[index, 1])
        # vehicle states by learning-based-model
        vehicle_state_fnn[index, :] = vehicle_state_fnn[index - 1, :] + \
            output_fnn[index, :] * DELTA_T
        vehicle_state_fnn[index, 1] = normalize_angle(vehicle_state_fnn[index, 1])
        # vehicle states by sim_point_mass
        vehicle_state_point_mass[index, :] = vehicle_state_point_mass[index - 1, :] + \
            output_point_mass[index, :] * DELTA_T
        vehicle_state_point_mass[index, 1] = normalize_angle(vehicle_state_point_mass[index, 1])

    rmse_imu_speed = sqrt(mean_squared_error(vehicle_state_imu[:, 0], vehicle_state_gps[:, 0]))
    rmse_fnn_speed = sqrt(mean_squared_error(vehicle_state_fnn[:, 0], vehicle_state_gps[:, 0]))
    rmse_point_mass_speed = sqrt(mean_squared_error(vehicle_state_point_mass[:, 0],
                                                    vehicle_state_gps[:, 0]))
    rms_speed = sqrt(sum(n * n for n in vehicle_state_gps[:, 0]) / len(vehicle_state_gps[:, 0]))

    evaluation_results.sensor_error.speed_error = rmse_imu_speed
    evaluation_results.sensor_error.speed_error_rate = rmse_imu_speed / rms_speed
    evaluation_results.learning_based_result.speed_error = rmse_fnn_speed
    evaluation_results.learning_based_result.speed_error_rate = rmse_fnn_speed / rms_speed
    evaluation_results.point_mass_result.speed_error = rmse_point_mass_speed
    evaluation_results.point_mass_result.speed_error_rate = rmse_point_mass_speed / rms_speed

    rmse_imu_heading = sqrt(mean_squared_error(vehicle_state_imu[:, 1], vehicle_state_gps[:, 1]))
    rmse_fnn_heading = sqrt(mean_squared_error(vehicle_state_fnn[:, 1], vehicle_state_gps[:, 1]))
    rmse_point_mass_heading = sqrt(mean_squared_error(vehicle_state_point_mass[:, 1],
                                                      vehicle_state_gps[:, 1]))
    rms_heading = sqrt(
        sum(n * n for n in vehicle_state_gps[:, 1]) / len(vehicle_state_gps[:, 1]))

    evaluation_results.sensor_error.speed_error = rmse_imu_heading
    evaluation_results.sensor_error.speed_error_rate = rmse_imu_heading / rms_heading
    evaluation_results.learning_based_result.speed_error = rmse_fnn_heading
    evaluation_results.learning_based_result.speed_error_rate = rmse_fnn_heading / rms_heading
    evaluation_results.point_mass_result.speed_error = rmse_point_mass_heading
    evaluation_results.point_mass_result.speed_error_rate = rmse_point_mass_heading / rms_heading

    return vehicle_state_echo_lincoln, vehicle_state_imu, vehicle_state_fnn, \
        vehicle_state_point_mass


def evaluate_trajectory(trajectory_gps, vehicle_state_gps, vehicle_state_echo_lincoln,
                        vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass,
                        evaluation_results):
    trajectory_gps2 = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_echo_lincoln = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_imu = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_fnn = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_point_mass = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_gps2[0, :] = trajectory_gps[0, :]
    trajectory_echo_lincoln[0, :] = trajectory_gps[0, :]
    trajectory_imu[0, :] = trajectory_gps[0, :]
    trajectory_fnn[0, :] = trajectory_gps[0, :]
    trajectory_point_mass[0, :] = trajectory_gps[0, :]
    trajectory_length = 0

    for index in range(1, trajectory_gps.shape[0]):
        trajectory_gps2[index, 0] = trajectory_gps2[index - 1, 0] + vehicle_state_gps[index, 0] * \
            np.cos(vehicle_state_gps[index, 1]) * DELTA_T
        trajectory_gps2[index, 1] = trajectory_gps2[index - 1, 1] + vehicle_state_gps[index, 0] * \
            np.sin(vehicle_state_gps[index, 1]) * DELTA_T
        trajectory_echo_lincoln[index, 0] = trajectory_echo_lincoln[index - 1, 0] + \
            vehicle_state_echo_lincoln[index, 0] * \
            np.cos(vehicle_state_echo_lincoln[index, 1]) * DELTA_T
        trajectory_echo_lincoln[index, 1] = trajectory_echo_lincoln[index - 1, 1] + \
            vehicle_state_echo_lincoln[index, 0] * \
            np.sin(vehicle_state_echo_lincoln[index, 1]) * DELTA_T
        trajectory_imu[index, 0] = trajectory_imu[index - 1, 0] + vehicle_state_imu[index, 0] * \
            np.cos(vehicle_state_imu[index, 1]) * DELTA_T
        trajectory_imu[index, 1] = trajectory_imu[index - 1, 1] + vehicle_state_imu[index, 0] * \
            np.sin(vehicle_state_imu[index, 1]) * DELTA_T
        trajectory_fnn[index, 0] = trajectory_fnn[index - 1, 0] + vehicle_state_fnn[index, 0] * \
            np.cos(vehicle_state_fnn[index, 1]) * DELTA_T
        trajectory_fnn[index, 1] = trajectory_fnn[index - 1, 1] + vehicle_state_fnn[index, 0] * \
            np.sin(vehicle_state_fnn[index, 1]) * DELTA_T
        trajectory_point_mass[index, 0] = trajectory_point_mass[index - 1, 0] + \
            vehicle_state_point_mass[index, 0] * \
            np.cos(vehicle_state_point_mass[index, 1]) * DELTA_T
        trajectory_point_mass[index, 1] = trajectory_point_mass[index - 1, 1] + \
            vehicle_state_point_mass[index, 0] * \
            np.sin(vehicle_state_point_mass[index, 1]) * DELTA_T
        trajectory_length += sqrt((trajectory_gps[index, 0] - trajectory_gps[index - 1, 0]) ** 2 + (
            trajectory_gps[index, 1] - trajectory_gps[index - 1, 1]) ** 2)

    rmse_imu_trajectory = sqrt(mean_squared_error(trajectory_imu, trajectory_gps))
    rmse_fnn_trajectory = sqrt(mean_squared_error(trajectory_fnn, trajectory_gps))
    rmse_point_mass_trajectory = sqrt(mean_squared_error(trajectory_point_mass, trajectory_gps))
    rmse_echo_lincoln_trajectory = sqrt(mean_squared_error(trajectory_echo_lincoln, trajectory_gps))

    evaluation_results.sensor_error.trajectory_error = rmse_imu_trajectory
    evaluation_results.sensor_error.trajectory_error_rate = \
        rmse_imu_trajectory / trajectory_length
    evaluation_results.learning_based_result.trajectory_error = rmse_fnn_trajectory
    evaluation_results.learning_based_result.trajectory_error_rate = \
        rmse_fnn_trajectory / trajectory_length
    evaluation_results.point_mass_result.trajectory_error = rmse_point_mass_trajectory
    evaluation_results.point_mass_result.trajectory_error_rate = \
        rmse_point_mass_trajectory / trajectory_length
    evaluation_results.echo_lincoln_result.trajectory_error = rmse_echo_lincoln_trajectory
    evaluation_results.echo_lincoln_result.trajectory_error_rate = \
        rmse_echo_lincoln_trajectory / trajectory_length
    return trajectory_gps2, trajectory_echo_lincoln, trajectory_imu, trajectory_fnn,\
        trajectory_point_mass


def visualize_evaluation_results(pdf_file_path, trajectory_gps, trajectory_gps2,
                                 trajectory_echo_lincoln, trajectory_imu, trajectory_fnn,
                                 trajectory_point_mass, vehicle_state_gps,
                                 vehicle_state_echo_lincoln, vehicle_state_imu, vehicle_state_fnn,
                                 vehicle_state_point_mass, output_echo_lincoln, output_imu,
                                 output_fnn):
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.title("Trajectory Visualization")
        # Plot the trajectory collected by GPS
        plt.plot(trajectory_gps[:, 0], trajectory_gps[:, 1], color='blue',
                 label="Ground-truth Tracjectory")
        plt.plot(trajectory_gps[-1, 0],
                 trajectory_gps[-1, 1], color='blue', marker='x')
        # Plot the trajectory collected by GPS2
        plt.plot(trajectory_gps2[:, 0], trajectory_gps2[:, 1], color='purple',
                 label="Ground-truth2 Tracjectory")
        plt.plot(trajectory_gps2[-1, 0],
                 trajectory_gps2[-1, 1], color='purple', marker='x')
        # Plot the trajectory calculated by Echo_lincoln
        plt.plot(trajectory_echo_lincoln[:, 0], trajectory_echo_lincoln[:, 1], color='black',
                 label="Echo_lincoln Tracjectory")
        plt.plot(trajectory_echo_lincoln[-1, 0],
                 trajectory_echo_lincoln[-1, 1], color='black', marker='x')
        # Plot the trajectory calculated by IMU
        plt.plot(trajectory_imu[:, 0], trajectory_imu[:, 1], color='orange', alpha=ALPHA,
                 label="Generated Tracjectory by IMU")
        plt.plot(trajectory_imu[-1, 0],
                 trajectory_imu[-1, 1], color='orange', alpha=ALPHA, marker='x')
        # Plot the trajectory calculated by learning-based model
        plt.plot(trajectory_fnn[:, 0], trajectory_fnn[:, 1], color='red',
                 label="Tracjectory by learning-based-model")
        plt.plot(trajectory_fnn[-1, 0], trajectory_fnn[-1, 1], color='red', marker='x')
        if not IS_BACKWARD:
            # Plot the trajectory calculated by point_mass model
            plt.plot(trajectory_point_mass[:, 0], trajectory_point_mass[:, 1], color='green',
                     label="Tracjectory by sim_point_mass")
            plt.plot(trajectory_point_mass[-1, 0], trajectory_point_mass[-1, 1],
                     color='green', marker='x')
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Plot the speed calculated by different models
        plt.figure(figsize=(4, 3))
        plt.title("Vehicle Speed Visualization")
        plt.plot(vehicle_state_gps[:, 0], color='blue', label="Ground-truth Speed")
        plt.plot(vehicle_state_echo_lincoln[:, 0], color='black', label="Echo_lincoln Speed")
        plt.plot(vehicle_state_imu[:, 0], color='orange', alpha=ALPHA, label="IMU Speed")
        plt.plot(vehicle_state_fnn[:, 0], color='red', label="FNN Speed")
        if not IS_BACKWARD:
            plt.plot(vehicle_state_point_mass[:, 0], color='green', label="PointMass Speed")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Plot the heading calculated by different models
        plt.figure(figsize=(4, 3))
        plt.title("Vehicle Heading Visualization")
        plt.plot(vehicle_state_gps[:, 1], color='blue', label="Ground-truth Heading")
        plt.plot(vehicle_state_echo_lincoln[:, 1], color='black', label="Echo_lincoln Heading")
        plt.plot(vehicle_state_imu[:, 1], color='orange', label="IMU Heading")
        plt.plot(vehicle_state_fnn[:, 1], color='red', label="FNN Heading")
        if not IS_BACKWARD:
            plt.plot(vehicle_state_point_mass[:, 1], color='green', label="PointMass Heading")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Plot the acceleration calculated by fnn and imu
        plt.figure(figsize=(4, 3))
        plt.title("Vehicle Acceleration Visualization")

        plt.plot(output_imu[:, 0], color='orange', alpha=ALPHA, label="Echo_lincoln Acceleration")
        plt.plot(output_echo_lincoln[:, 0], color='black', label="IMU Acceleration")
        plt.plot(output_fnn[:, 0], color='red', label="FNN Acceleration")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Plot the angular velocity calculated by fnn and imu
        plt.figure(figsize=(4, 3))
        plt.title("Vehicle Angular Speed Visualization")
        plt.plot(output_imu[:, 1], color='orange', alpha=ALPHA, label="IMU Angular Speed")
        # if PLOT_MODEL:
        plt.plot(output_echo_lincoln[:, 1], color='black', label="Echo_lincoln Angular Speed")
        plt.plot(output_fnn[:, 1], color='red', label="FNN Angular Speed")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()


def evaluate(model_info, dataset_info, platform_path):
    if model_info[0] == 'mlp':
        vehicle_state_gps, output_echo_lincoln, output_imu, output_point_mass, output_fnn, \
            trajectory_gps = data_generator.generate_evaluation_data(dataset_info[1],
                                                                     model_info[1], 'mlp')
    elif model_info[0] == 'lstm':
        vehicle_state_gps, output_echo_lincoln, output_imu, output_point_mass, output_fnn, \
            trajectory_gps = data_generator.generate_evaluation_data(dataset_info[1],
                                                                     model_info[1], 'lstm')
    else:
        return

    # Dump the quantitative evaluation results to a protobuf-format txt file
    evaluation_results = EvaluationResults()
    evaluation_result_path = os.path.join(model_info[1],
                                          'evaluation_metrics_under_%s.txt' % dataset_info[0])
    # Evaluate the accuracy of direct outputs of dynamic models
    evaluate_direct_output(output_imu, output_fnn,
                           output_point_mass, evaluation_results)

    # Evaluate the accuracy of vehicle states by first integration over time
    vehicle_state_echo_lincoln, vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass = \
        evaluate_vehicle_state(vehicle_state_gps, output_echo_lincoln, output_imu, output_fnn,
                               output_point_mass, evaluation_results)

    trajectory_gps2, trajectory_echo_lincoln, trajectory_imu, trajectory_fnn,\
        trajectory_point_mass = evaluate_trajectory(trajectory_gps, vehicle_state_gps,
                                                    vehicle_state_echo_lincoln,
                                                    vehicle_state_imu, vehicle_state_fnn,
                                                    vehicle_state_point_mass,
                                                    evaluation_results)

    with open(evaluation_result_path, 'w') as txt_file:
        txt_file.write('evaluated on model: {} \n'.format(model_info[1]))
        txt_file.write('evaluated on record: {} \n'.format(dataset_info[1]))
        txt_file.write(text_format.MessageToString(evaluation_results))

    # Output the trajectory visualization plots to a pdf file
    pdf_file_path = os.path.join(model_info[1],
                                 'trajectory_visualization_under_%s.pdf' % dataset_info[0])
    logging.info('pdf_file_path: {}'.format(pdf_file_path))
    visualize_evaluation_results(pdf_file_path, trajectory_gps, trajectory_gps2,
                                 trajectory_echo_lincoln, trajectory_imu, trajectory_fnn,
                                 trajectory_point_mass, vehicle_state_gps,
                                 vehicle_state_echo_lincoln, vehicle_state_imu, vehicle_state_fnn,
                                 vehicle_state_point_mass, output_echo_lincoln, output_imu,
                                 output_fnn)

    # return (dynamic_model_path, Trajectory_RMSE)
    return [(model_info[1], evaluation_results.learning_based_result.trajectory_error)]
