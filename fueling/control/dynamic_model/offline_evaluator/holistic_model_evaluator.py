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
import colored_glog as glog
import matplotlib.pyplot as plt
import numpy as np

from fueling.control.dynamic_model.conf.model_config import feature_config
from modules.data.fuel.fueling.control.proto.dynamic_model_evaluation_pb2 import EvaluationResults
import fueling.control.dynamic_model.data_generator.holistic_data_generator as data_generator

# System setup
USE_TENSORFLOW = True  # Slightly faster than Theano.
USE_GPU = False  # CPU seems to be faster than GPU in this case.

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
DELTA_T = feature_config["delta_t"]


def evaluate_direct_output(output_imu, output_fnn, output_point_mass, evaluation_results):

    rmse_fnn_acceleration = sqrt(mean_squared_error(
        output_imu[:, 0], output_fnn[:, 0]))
    rmse_point_mass_acceleration = sqrt(mean_squared_error(
        output_imu[:, 0], output_point_mass[:, 0]))
    rms_acceleration = sqrt(
        sum(n * n for n in output_imu[:, 0]) / len(output_imu[:, 0]))
    evaluation_results.learning_based_result.acceleration_error = rmse_fnn_acceleration
    evaluation_results.learning_based_result.acceleration_error_rate = \
        rmse_fnn_acceleration / rms_acceleration
    evaluation_results.point_mass_result.acceleration_error = rmse_point_mass_acceleration
    evaluation_results.point_mass_result.acceleration_error_rate = \
        rmse_point_mass_acceleration / rms_acceleration

    rmse_fnn_angular_speed = sqrt(
        mean_squared_error(output_imu[:, 1], output_fnn[:, 1]))
    rmse_point_mass_angular_speed = sqrt(
        mean_squared_error(output_imu[:, 1], output_point_mass[:, 1]))
    rms_angular_speed = sqrt(
        sum(n * n for n in output_imu[:, 1]) / len(output_imu[:, 1]))
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


def evaluate_vehicle_state(vehicle_state_gps, output_imu, output_fnn, output_point_mass,
                           evaluation_results):
    vehicle_state_imu = np.zeros(
        [vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_fnn = np.zeros(
        [vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_point_mass = np.zeros(
        [vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_imu[0, :] = vehicle_state_gps[0, :]
    vehicle_state_fnn[0, :] = vehicle_state_gps[0, :]
    vehicle_state_point_mass[0, :] = vehicle_state_gps[0, :]

    for k in range(1, vehicle_state_gps.shape[0]):
        # vehicle states by imu sensor
        vehicle_state_imu[k, :] = vehicle_state_imu[k -
                                                    1, :] + output_imu[k, :] * DELTA_T
        vehicle_state_imu[k, 1] = normalize_angle(vehicle_state_imu[k, 1])
        # vehicle states by learning-based-model
        vehicle_state_fnn[k, :] = vehicle_state_fnn[k -
                                                    1, :] + output_fnn[k, :] * DELTA_T
        vehicle_state_fnn[k, 1] = normalize_angle(vehicle_state_fnn[k, 1])
        # vehicle states by sim_point_mass
        vehicle_state_point_mass[k, :] = vehicle_state_point_mass[k - 1, :] + \
            output_point_mass[k, :] * DELTA_T
        vehicle_state_point_mass[k, 1] = normalize_angle(
            vehicle_state_point_mass[k, 1])

    rmse_imu_lon_speed = sqrt(mean_squared_error(
        vehicle_state_imu[:, 0], vehicle_state_gps[:, 0]))
    rmse_fnn_lon_speed = sqrt(mean_squared_error(
        vehicle_state_fnn[:, 0], vehicle_state_gps[:, 0]))
    rmse_point_mass_lon_speed = sqrt(mean_squared_error(vehicle_state_point_mass[:, 0],
                                                        vehicle_state_gps[:, 0]))
    rms_lon_speed = sqrt(
        sum(n * n for n in vehicle_state_gps[:, 0]) / len(vehicle_state_gps[:, 0]))

    evaluation_results.sensor_error.speed_error = rmse_imu_lon_speed
    evaluation_results.sensor_error.speed_error_rate = rmse_imu_lon_speed / rms_lon_speed
    evaluation_results.learning_based_result.speed_error = rmse_fnn_lon_speed
    evaluation_results.learning_based_result.speed_error_rate = rmse_fnn_lon_speed / rms_lon_speed
    evaluation_results.point_mass_result.speed_error = rmse_point_mass_lon_speed
    evaluation_results.point_mass_result.speed_error_rate = rmse_point_mass_lon_speed / rms_lon_speed

    rmse_imu_lat_speed = sqrt(mean_squared_error(
        vehicle_state_imu[:, 1], vehicle_state_gps[:, 1]))
    rmse_fnn_lat_speed = sqrt(mean_squared_error(
        vehicle_state_fnn[:, 1], vehicle_state_gps[:, 1]))
    rmse_point_mass_lat_speed = sqrt(mean_squared_error(vehicle_state_point_mass[:, 1],
                                                        vehicle_state_gps[:, 1]))
    rms_lat_speed = sqrt(
        sum(n * n for n in vehicle_state_gps[:, 1]) / len(vehicle_state_gps[:, 1]))

    evaluation_results.sensor_error.speed_error = rmse_imu_lat_speed
    evaluation_results.sensor_error.speed_error_rate = rmse_imu_lat_speed / rms_lat_speed
    evaluation_results.learning_based_result.speed_error = rmse_fnn_lat_speed
    evaluation_results.learning_based_result.speed_error_rate = rmse_fnn_lat_speed / rms_lat_speed
    evaluation_results.point_mass_result.speed_error = rmse_point_mass_lat_speed
    evaluation_results.point_mass_result.speed_error_rate = rmse_point_mass_lat_speed / rms_lat_speed

    rmse_imu_heading = sqrt(mean_squared_error(
        vehicle_state_imu[:, 2], vehicle_state_gps[:, 2]))
    rmse_fnn_heading = sqrt(mean_squared_error(
        vehicle_state_fnn[:, 2], vehicle_state_gps[:, 2]))
    rmse_point_mass_heading = sqrt(mean_squared_error(vehicle_state_point_mass[:, 2],
                                                      vehicle_state_gps[:, 2]))
    rms_heading = sqrt(
        sum(n * n for n in vehicle_state_gps[:, 2]) / len(vehicle_state_gps[:, 2]))

    evaluation_results.sensor_error.speed_error = rmse_imu_heading
    evaluation_results.sensor_error.speed_error_rate = rmse_imu_heading / rms_heading
    evaluation_results.learning_based_result.speed_error = rmse_fnn_heading
    evaluation_results.learning_based_result.speed_error_rate = rmse_fnn_heading / rms_heading
    evaluation_results.point_mass_result.speed_error = rmse_point_mass_heading
    evaluation_results.point_mass_result.speed_error_rate = rmse_point_mass_heading / rms_heading

    return vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass


def evaluate_trajectory(trajectory_gps, vehicle_state_imu, vehicle_state_fnn,
                        vehicle_state_point_mass, evaluation_results):
    trajectory_imu = np.zeros(
        [trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_fnn = np.zeros(
        [trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_point_mass = np.zeros(
        [trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_imu[0, :] = trajectory_gps[0, :]
    trajectory_fnn[0, :] = trajectory_gps[0, :]
    trajectory_point_mass[0, :] = trajectory_gps[0, :]
    trajectory_length = 0

    for k in range(1, trajectory_gps.shape[0]):
        trajectory_imu[k, 0] = trajectory_imu[k - 1, 0] + vehicle_state_imu[k, 0] * \
            np.cos(vehicle_state_imu[k, 2]) * DELTA_T + \
            vehicle_state_imu[k, 1] * np.sin(vehicle_state_imu[k, 2]) * DELTA_T
        trajectory_imu[k, 1] = trajectory_imu[k - 1, 1] + vehicle_state_imu[k, 0] * \
            np.sin(vehicle_state_imu[k, 2]) * DELTA_T - \
            vehicle_state_imu[k, 1] * np.cos(vehicle_state_imu[k, 2]) * DELTA_T

        trajectory_fnn[k, 0] = trajectory_fnn[k - 1, 0] + vehicle_state_fnn[k, 0] * \
            np.cos(vehicle_state_fnn[k, 2]) * DELTA_T + \
            vehicle_state_fnn[k, 1] * np.sin(vehicle_state_fnn[k, 2]) * DELTA_T
        trajectory_fnn[k, 1] = trajectory_fnn[k - 1, 1] + vehicle_state_fnn[k, 0] * \
            np.sin(vehicle_state_fnn[k, 2]) * DELTA_T - \
            vehicle_state_fnn[k, 1] * np.cos(vehicle_state_fnn[k, 2]) * DELTA_T

        trajectory_point_mass[k, 0] = trajectory_point_mass[k - 1, 0] + \
            vehicle_state_point_mass[k, 0] * np.cos(vehicle_state_point_mass[k, 2]) * DELTA_T + \
            vehicle_state_point_mass[k, 1] * \
            np.sin(vehicle_state_point_mass[k, 2]) * DELTA_T
        trajectory_point_mass[k, 1] = trajectory_point_mass[k - 1, 1] + \
            vehicle_state_point_mass[k, 0] * np.sin(vehicle_state_point_mass[k, 2]) * DELTA_T - \
            vehicle_state_point_mass[k, 1] * \
            np.cos(vehicle_state_point_mass[k, 2]) * DELTA_T

        trajectory_length += sqrt((trajectory_gps[k, 0] - trajectory_gps[k - 1, 0]) ** 2 + (
            trajectory_gps[k, 1] - trajectory_gps[k - 1, 1]) ** 2)

    rmse_imu_trajectory = sqrt(
        mean_squared_error(trajectory_imu, trajectory_gps))
    rmse_fnn_trajectory = sqrt(
        mean_squared_error(trajectory_fnn, trajectory_gps))
    rmse_point_mass_trajectory = sqrt(
        mean_squared_error(trajectory_point_mass, trajectory_gps))

    evaluation_results.sensor_error.trajectory_error = rmse_imu_trajectory
    evaluation_results.sensor_error.trajectory_error_rate = \
        rmse_imu_trajectory / trajectory_length
    evaluation_results.learning_based_result.trajectory_error = rmse_fnn_trajectory
    evaluation_results.learning_based_result.trajectory_error_rate = \
        rmse_fnn_trajectory / trajectory_length
    evaluation_results.point_mass_result.trajectory_error = rmse_point_mass_trajectory
    evaluation_results.point_mass_result.trajectory_error_rate = \
        rmse_point_mass_trajectory / trajectory_length
    return trajectory_imu, trajectory_fnn, trajectory_point_mass


def visualize_evaluation_results(pdf_file_path, trajectory_gps, trajectory_imu, trajectory_fnn,
                                 trajectory_point_mass, vehicle_state_gps, vehicle_state_imu,
                                 vehicle_state_fnn, vehicle_state_point_mass):
    with PdfPages(pdf_file_path) as pdf_file:
        plt.figure(figsize=(4, 3))
        plt.title("Trajectory Visualization")
        # Plot the trajectory collected by GPS
        plt.plot(trajectory_gps[:, 0], trajectory_gps[:, 1], color='blue',
                 label="Ground-truth Tracjectory")
        plt.plot(trajectory_gps[-1, 0],
                 trajectory_gps[-1, 1], color='blue', marker='x')
        # Plot the trajectory calculated by IMU
        plt.plot(trajectory_imu[:, 0], trajectory_imu[:, 1], color='orange',
                 label="Generated Tracjectory by IMU")
        plt.plot(trajectory_imu[-1, 0],
                 trajectory_imu[-1, 1], color='orange', marker='x')
        # Plot the trajectory calculated by learning-based model
        plt.plot(trajectory_fnn[:, 0], trajectory_fnn[:, 1], color='red',
                 label="Tracjectory by learning-based-model")
        plt.plot(trajectory_fnn[-1, 0],
                 trajectory_fnn[-1, 1], color='red', marker='x')
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
        plt.plot(vehicle_state_gps[:, 0],
                 color='blue', label="Ground-truth Speed")
        plt.plot(vehicle_state_imu[:, 0], color='orange', label="IMU Speed")
        plt.plot(vehicle_state_point_mass[:, 0],
                 color='green', label="PointMass Speed")
        plt.plot(vehicle_state_fnn[:, 0], color='red', label="FNN Speed")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()

        # Plot the heading calculated by different models
        plt.figure(figsize=(4, 3))
        plt.title("Vehicle Heading Visualization")
        plt.plot(vehicle_state_gps[:, 1],
                 color='blue', label="Ground-truth Heading")
        plt.plot(vehicle_state_imu[:, 1], color='orange', label="IMU Heading")
        plt.plot(vehicle_state_point_mass[:, 1],
                 color='green', label="PointMass Heading")
        plt.plot(vehicle_state_fnn[:, 1], color='red', label="FNN Heading")
        plt.legend()
        pdf_file.savefig()  # saves the current figure into a pdf page
        plt.close()


def evaluate(model_info, dataset_info, platform_path):
    if model_info[0] == 'mlp':
        vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps = \
            data_generator.generate_evaluation_data(
                dataset_info[1], model_info[1], 'mlp')
    elif model_info[0] == 'lstm':
        vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps = \
            data_generator.generate_evaluation_data(
                dataset_info[1], model_info[1], 'lstm')
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
    vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass = evaluate_vehicle_state(
        vehicle_state_gps, output_imu, output_fnn, output_point_mass, evaluation_results)
    trajectory_imu, trajectory_fnn, trajectory_point_mass = evaluate_trajectory(trajectory_gps,
                                                                                vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass, evaluation_results)
    with open(evaluation_result_path, 'w') as txt_file:
        txt_file.write('evaluted on model: {} \n'.format(model_info[1]))
        txt_file.write('evaluted on record: {} \n'.format(dataset_info[1]))
        txt_file.write(text_format.MessageToString(evaluation_results))

    # Output the trajectory visualization plots to a pdf file
    pdf_file_path = os.path.join(model_info[1],
                                 'trajectory_visualization_under_%s.pdf' % dataset_info[0])
    visualize_evaluation_results(pdf_file_path, trajectory_gps, trajectory_imu, trajectory_fnn,
                                 trajectory_point_mass, vehicle_state_gps, vehicle_state_imu,
                                 vehicle_state_fnn, vehicle_state_point_mass)

    # return (dynamic_model_path, Trajectory_RMSE)
    return [(model_info[1], evaluation_results.learning_based_result.trajectory_error)]
