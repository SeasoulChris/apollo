#!/usr/bin/env python

from math import sqrt
import math
import os
import sys
import time

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error
import numpy as np

from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.colored_glog as glog
import fueling.control.dynamic_model.data_generator.data_generator as data_generator

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
        os.environ["THEANORC"] = os.path.join(os.getcwd(), "theanorc/gpu_config")
        os.environ["DEVICE"] = "cuda"  # for pygpu, unclear whether necessary
    else:
        os.environ["THEANORC"] = os.path.join(os.getcwd(), "theanorc/cpu_config")

# Constants
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]
DIM_LSTM_LENGTH = feature_config["sequence_length"]
DELTA_T = feature_config["delta_t"]


def plot_model_output(output_imu, output_fnn, output_point_mass, txt, *pdf):

    # plt.figure(figsize=(3,2))
    # plt.title("Acceleration")
    # plt.plot(output_imu[:,0], color = 'blue',label = "IMU acceleration")
    # plt.plot(output_fnn[:,0], color = 'red', label = "Acceleration by learning-based-model")
    # plt.plot(output_point_mass[:,0], color = 'green',label = "Acceleration by sim_point_mass")
    rmse_fnn_acceleration = sqrt(mean_squared_error(output_imu[:, 0], output_fnn[:, 0]))
    rmse_point_mass_acceleration = sqrt(mean_squared_error(output_imu[:, 0], output_point_mass[:, 0]))
    rms_acceleration = sqrt(sum(n * n for n in output_imu[:, 0]) / len(output_imu[:, 0]))
    txt.write("Acceleration rmse of learning-based-model: {}, Error Rate: {}".format(
        rmse_fnn_acceleration, rmse_fnn_acceleration / rms_acceleration))
    txt.write("\n")
    txt.write("Acceleration rmse of sim_point_mass: {}, Error Rate: {}".format(
        rmse_point_mass_acceleration, rmse_point_mass_acceleration / rms_acceleration))
    txt.write("\n")
    # pdf.savefig()  # saves the current figure into a pdf page
    # plt.close()

    # plt.figure(figsize=(3,2))
    # plt.title("Angular speed")
    # plt.plot(output_imu[:,1], color = 'blue',label = "IMU angular speed")
    # plt.plot(output_fnn[:,1], color = 'red', label = "Angular speed by learning-based-model")
    # plt.plot(output_point_mass[:,1], color = 'green',label = "Angular speed by sim_point_mass")
    rmse_fnn_angular_speed = sqrt(mean_squared_error(output_imu[:, 1], output_fnn[:, 1]))
    rmse_point_mass_angular_speed = sqrt(mean_squared_error(output_imu[:, 1], output_point_mass[:, 1]))
    rms_angular_speed = sqrt(sum(n * n for n in output_imu[:, 1]) / len(output_imu[:, 1]))
    txt.write("Angular speed rmse of learning-based-model: {}, Error Rate: {}".format(
        rmse_fnn_angular_speed, rmse_fnn_angular_speed / rms_angular_speed))
    txt.write("\n")
    txt.write("Angular speed rmse of sim_point_mass: {}, Error Rate: {}".format(
        rmse_point_mass_angular_speed, rmse_point_mass_angular_speed / rms_angular_speed))
    txt.write("\n")
    # pdf.savefig()  # saves the current figure into a pdf page
    # plt.close()


def normalize_angle(theta):
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta = theta - 2 * math.pi
    return theta


def plot_first_integral(vehicle_state_gps, output_imu, output_fnn, output_point_mass, txt, *pdf):
    vehicle_state_imu = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_fnn = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_point_mass = np.zeros([vehicle_state_gps.shape[0], vehicle_state_gps.shape[1]])
    vehicle_state_imu[0, :] = vehicle_state_gps[0, :]
    vehicle_state_fnn[0, :] = vehicle_state_gps[0, :]
    vehicle_state_point_mass[0, :] = vehicle_state_gps[0, :]

    for k in range(1, vehicle_state_gps.shape[0]):
        vehicle_state_imu[k, :] = vehicle_state_imu[k - 1, :] + \
                                    output_imu[k, :] * DELTA_T  # by imu sensor
        vehicle_state_imu[k, 1] = normalize_angle(vehicle_state_imu[k, 1])
        vehicle_state_fnn[k, :] = vehicle_state_fnn[k - 1, :] + \
                                    output_fnn[k, :] * DELTA_T  # by learning-based-model
        vehicle_state_fnn[k, 1] = normalize_angle(vehicle_state_fnn[k, 1])
        vehicle_state_point_mass[k, :] = vehicle_state_point_mass[k - 1, :] + \
                                            output_point_mass[k, :] * DELTA_T  # by sim_point_mass
        vehicle_state_point_mass[k, 1] = normalize_angle(vehicle_state_point_mass[k, 1])

    rmse_imu_speed = sqrt(mean_squared_error(vehicle_state_imu[:, 0], vehicle_state_gps[:, 0]))
    rmse_fnn_speed = sqrt(mean_squared_error(vehicle_state_fnn[:, 0], vehicle_state_gps[:, 0]))
    rmse_point_mass_speed = sqrt(mean_squared_error(vehicle_state_point_mass[:, 0], 
                                                        vehicle_state_gps[:, 0]))
    rms_speed = sqrt(sum(n * n for n in vehicle_state_gps[:, 0]) / len(vehicle_state_gps[:, 0]))

    # plt.figure(figsize=(3,2))
    # plt.title("Speed")
    # plt.plot(vehicle_state_gps[:,0], color = 'blue',label = "Ground-truth speed")
    # plt.plot(vehicle_state_imu[:,0], color = 'orange',label = "Incremental speed by sensors")
    # plt.plot(vehicle_state_fnn[:,0], color = 'red', label = "Speed by learning-based-model")
    # plt.plot(vehicle_state_point_mass[:,0], color = 'green',label = "Speed by sim_point_mass")

    txt.write("Speed rmse of sensor: {}, Error Rate: {}".format(
        rmse_imu_speed, rmse_imu_speed / rms_speed))
    txt.write("\n")
    txt.write("Speed rmse of learning-based-model: {}, Error Rate: {}".format(
        rmse_fnn_speed, rmse_fnn_speed / rms_speed))
    txt.write("\n")
    txt.write("Speed rmse of sim_point_mass: {}, Error Rate: {}".format(
        rmse_point_mass_speed, rmse_point_mass_speed / rms_speed))
    txt.write("\n")
    # pdf.savefig()  # saves the current figure into a pdf page
    # plt.close()

    rmse_imu_heading = sqrt(mean_squared_error(vehicle_state_imu[:, 1], vehicle_state_gps[:, 1]))
    rmse_fnn_heading = sqrt(mean_squared_error(vehicle_state_fnn[:, 1], vehicle_state_gps[:, 1]))
    rmse_point_mass_heading = sqrt(mean_squared_error(vehicle_state_point_mass[:, 1], 
                                    vehicle_state_gps[:, 1]))
    rms_heading = sqrt(sum(n * n for n in vehicle_state_gps[:, 1]) / len(vehicle_state_gps[:, 1]))

    # plt.figure(figsize=(3,2))
    # plt.title("Heading")
    # plt.plot(vehicle_state_gps[:,1], color = 'blue', label = "Ground-truth heading")
    # plt.plot(vehicle_state_imu[:,1], color = 'orange', label = "Incremental heading by angular speed")
    # plt.plot(vehicle_state_fnn[:,1], color = 'red', label = "Heading by learning-based-model")
    # plt.plot(vehicle_state_point_mass[:,1], color = 'green', label = "Heading by sim_point_mass")

    txt.write("Heading rmse of sensor: {}, Error Rate: {}".format(
        rmse_imu_heading, rmse_imu_heading / rms_heading))
    txt.write("\n")
    txt.write("Heading rmse of learning-based-model: {}, Error Rate: {}".format(
        rmse_fnn_heading, rmse_fnn_heading / rms_heading))
    txt.write("\n")
    txt.write("Heading rmse of sim_point_mass: {}, Error Rate: {}".format(
        rmse_point_mass_heading, rmse_point_mass_heading / rms_heading))
    txt.write("\n")
    # pdf.savefig()  # saves the current figure into a pdf page
    # plt.close()
    return vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass


def plot_trajectory(trajectory_gps, 
                    vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass, txt, *pdf):
    trajectory_imu = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_fnn = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_point_mass = np.zeros([trajectory_gps.shape[0], trajectory_gps.shape[1]])
    trajectory_imu[0, :] = trajectory_gps[0, :]
    trajectory_fnn[0, :] = trajectory_gps[0, :]
    trajectory_point_mass[0, :] = trajectory_gps[0, :]
    Trajectory_length = 0

    for k in range(1, trajectory_gps.shape[0]):
        trajectory_imu[k, 0] = trajectory_imu[k - 1, 0] + vehicle_state_imu[k, 0] * \
                                np.cos(vehicle_state_imu[k, 1]) * DELTA_T
        trajectory_imu[k, 1] = trajectory_imu[k - 1, 1] + vehicle_state_imu[k, 0] * \
                                np.sin(vehicle_state_imu[k, 1]) * DELTA_T
        trajectory_fnn[k, 0] = trajectory_fnn[k - 1, 0] + vehicle_state_fnn[k, 0] * \
                                np.cos(vehicle_state_fnn[k, 1]) * DELTA_T
        trajectory_fnn[k, 1] = trajectory_fnn[k - 1, 1] + vehicle_state_fnn[k, 0] * \
                                np.sin(vehicle_state_fnn[k, 1]) * DELTA_T
        trajectory_point_mass[k, 0] = trajectory_point_mass[k - 1, 0] + \
                vehicle_state_point_mass[k, 0] * np.cos(vehicle_state_point_mass[k, 1]) * DELTA_T
        trajectory_point_mass[k, 1] = trajectory_point_mass[k - 1, 1] + \
                vehicle_state_point_mass[k, 0] * np.sin(vehicle_state_point_mass[k, 1]) * DELTA_T
        Trajectory_length += np.sqrt((trajectory_gps[k, 0] - trajectory_gps[k - 1, 0]) ** 2 + (
                                        trajectory_gps[k, 1] - trajectory_gps[k - 1, 1]) ** 2)

    # fig = plt.figure(figsize=(3,2))
    # plt.title("Trajectory")
    # plt.plot(trajectory_gps[:,0], trajectory_gps[:,1], color = 'blue', label = "Ground-truth Tracjectory")
    # plt.plot(trajectory_gps[-1,0], trajectory_gps[-1,1], color = 'blue', marker = 'x')
    # plt.plot(trajectory_imu[:,0], trajectory_imu[:,1], color = 'orange', label = "Generated Tracjectory by IMU")
    # plt.plot(trajectory_imu[-1,0], trajectory_imu[-1,1], color = 'orange', marker = 'x')
    # plt.plot(trajectory_fnn[:,0], trajectory_fnn[:,1], color = 'red', label = "Tracjectory by learning-based-model")
    # plt.plot(trajectory_fnn[-1,0], trajectory_fnn[-1,1],color = 'red', marker = 'x')
    # plt.plot(trajectory_point_mass[:,0], trajectory_point_mass[:,1], color = 'green',label = "Tracjectory by sim_point_mass")
    # plt.plot(trajectory_point_mass[-1,0], trajectory_point_mass[-1,1], color = 'green', marker = 'x')

    rmse_imu_trajectory = sqrt(mean_squared_error(trajectory_imu, trajectory_gps))
    rmse_fnn_trajectory = sqrt(mean_squared_error(trajectory_fnn, trajectory_gps))
    rmse_point_mass_trajectory = sqrt(mean_squared_error(trajectory_point_mass, trajectory_gps))

    txt.write("Trajectory rmse of sensor: {}, Error Rate: {}".format(
        rmse_imu_trajectory, rmse_imu_trajectory / Trajectory_length))
    txt.write("\n")
    txt.write("Trajectory rmse of learning-based-model: {}, Error Rate: {}".format(
        rmse_fnn_trajectory, rmse_fnn_trajectory / Trajectory_length))
    txt.write("\n")
    txt.write("Trajectory rmse of sim_point_mass: {}, Error Rate: {}".format(
        rmse_point_mass_trajectory, rmse_point_mass_trajectory / Trajectory_length))
    txt.write("\n")
    # pdf.savefig()  # saves the current figure into a pdf page
    # plt.close()


def evaluate(model_info, dataset_path, platform_path):
    if model_info[0] == 'mlp':
        vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps = \
                    data_generator.generate_evaluation_data(dataset_path, model_info[1], 'mlp')
    elif model_info[0] == 'lstm':
        vehicle_state_gps, output_imu, output_point_mass, output_fnn, trajectory_gps = \
                    data_generator.generate_evaluation_data(dataset_path, model_info[1], 'lstm')
    else:
        return

    evaluation_result_path = os.path.join(platform_path, 
        'evaluation_result/evaluation_metrics_for_' + model_info[0] + '_model.txt')
    with open(evaluation_result_path, 'a') as txt:
        txt.write('\n evaluted on model: {} \n'.format(model_info[1]))
        txt.write('evaluted on record: {} \n'.format(dataset_path))
        plot_model_output(output_imu, output_fnn, output_point_mass, txt)
        vehicle_state_imu, vehicle_state_fnn, vehicle_state_point_mass = plot_first_integral(
            vehicle_state_gps, output_imu, output_fnn, output_point_mass, txt)
        plot_trajectory(trajectory_gps, vehicle_state_imu, 
                            vehicle_state_fnn, vehicle_state_point_mass, txt)
