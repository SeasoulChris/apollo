#!/usr/bin/env python

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import os
import glob
import numpy as np
import sys
import time
import math
from math import sqrt
from random import choice
from random import randint
from random import shuffle

import h5py
import google.protobuf.text_format as text_format
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.callbacks import ModelCheckpoint
from keras.metrics import mse
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils
from keras.regularizers import l2, l1
from keras.models import model_from_json
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.signal import savgol_filter

from fueling.control.features.parameters_training import dim
from fueling.control.lib.proto.fnn_model_pb2 import FnnModel, Layer

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
dim_input = dim["pose"] + dim["incremental"] + dim["control"] # accounts for mps
dim_output = dim["incremental"] # the speed mps is also output
input_features = ["speed mps","speed incremental","angular incremental","throttle","brake","steering"]


def generate_evaluation_data(segments):
    total_len = 0
    for i in range(len(segments)):
        total_len += (segments[i].shape[0] - 2)
    print "total_len = ", total_len
    X = np.zeros([total_len, dim_input])
    I_ground_truth = np.zeros([total_len, dim_output])
    Y_imu = np.zeros([total_len, dim_output])
    Y_point_mass = np.zeros([total_len, dim_output])
    T_ground_truth = np.zeros([total_len, dim_output])    
    print "X size = ", X.shape
    print "Y size = ", Y_imu.shape

    text_file = open("fueling/control/conf/sim_control_lincoln.pb.txt", "r")
    lines = text_file.readlines()
    table_length = len(lines)/5
    calibration_table = np.zeros([table_length, 3])
    for k in range(len(lines)):
        if k%5 != 0 and k%5 != 4:
            calibration_table[k/5,k%5-1] = float (lines[k].split(':')[1])
    
    f = interpolate.interp2d(calibration_table[:,0], calibration_table[:,1], calibration_table[:,2], kind='linear')

    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(segment.shape[0] - 1):
            if k>0:
                X[i,0] = segment[k-1,14] #speed mps
                X[i,1] = segment[k-1,8] * np.cos(segment[k-1,0]) + segment[k-1,9] * np.sin(segment[k-1,0])
                X[i,2] = segment[k-1,13]
                X[i,3] = segment[k-1,15] #throttle control from chassis 
                X[i,4] = segment[k-1,16] #brake control from chassis 
                X[i,5] = segment[k-1,17] #steering control from chassis 

                I_ground_truth[i,0] = segment[k,14] #speed_mps
                I_ground_truth[i,1] = segment[k,0]  #heading

                Y_imu[i,0] = segment[k,8] * np.cos(segment[k,0]) + segment[k,9] * np.sin(segment[k,0])
                Y_imu[i,1] = segment[k,13]
                if segment[k-1,15]-0.15141527 > segment[k-1,16]- 0.13727017:
                    lon_cmd = segment[k-1,15]
                else:
                    lon_cmd = -segment[k-1,16]
                Y_point_mass[i,0] = f(segment[k-1,14],lon_cmd* 100.0)
                Y_point_mass[i,1] = segment[k-1,17] * 8.203 /16 * segment[k-1,14] /2.8448

                T_ground_truth[i,0] = segment[k,19] 
                T_ground_truth[i,1] = segment[k,20] 
                i += 1

    return X, I_ground_truth, Y_imu, Y_point_mass, T_ground_truth

def plot_model_output(Y_imu, Y_mlp, Y_point_mass, pdf, txt):
    Y_imu[:,0] = savgol_filter(Y_imu[:,0], 51, 3) # window size 51, polynomial order 3
    Y_mlp[:,0] = savgol_filter(Y_mlp[:,0], 51, 3) # window size 51, polynomial order 3
    #plt.figure(figsize=(3,2))
    #plt.title("Acceleration")
    #plt.plot(Y_imu[:,0], color = 'blue',label = "IMU acceleration")
    #plt.plot(Y_mlp[:,0], color = 'red', label = "Acceleration by MLP")
    #plt.plot(Y_point_mass[:,0], color = 'green',label = "Acceleration by sim_point_mass")
    RMSE_mlp_acceleration = sqrt(mean_squared_error(Y_imu[:,0],Y_mlp[:,0]))
    RMSE_point_mass_acceleration = sqrt(mean_squared_error(Y_imu[:,0],Y_point_mass[:,0]))
    RMS_acceleration = sqrt(sum(n*n for n in Y_imu[:,0])/len(Y_imu[:,0]))
    txt.write("Acceleration RMSE of MLP:"+ str(RMSE_mlp_acceleration) + ", Error Rate:"+ str(RMSE_mlp_acceleration/RMS_acceleration))
    txt.write("\n")
    txt.write("Acceleration RMSE of sim_point_mass:"+ str(RMSE_point_mass_acceleration) + ", Error Rate:" + str(RMSE_point_mass_acceleration/RMS_acceleration))
    txt.write("\n")
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()

    #plt.figure(figsize=(3,2))
    #plt.title("Angular speed")
    #plt.plot(Y_imu[:,1], color = 'blue',label = "IMU angular speed")
    #plt.plot(Y_mlp[:,1], color = 'red', label = "Angular speed by MLP")
    #plt.plot(Y_point_mass[:,1], color = 'green',label = "Angular speed by sim_point_mass")
    RMSE_mlp_angular_speed = sqrt(mean_squared_error(Y_imu[:,1],Y_mlp[:,1]))
    RMSE_point_mass_angular_speed = sqrt(mean_squared_error(Y_imu[:,1],Y_point_mass[:,1]))
    RMS_angular_speed = sqrt(sum(n*n for n in Y_imu[:,1])/len(Y_imu[:,1]))
    txt.write( "Angular speed RMSE of MLP:" + str(RMSE_mlp_angular_speed) + ", Error Rate:" + str(RMSE_mlp_angular_speed/RMS_angular_speed))
    txt.write("\n")
    txt.write( "Angular speed RMSE of sim_point_mass:" + str(RMSE_point_mass_angular_speed) + ", Error Rate:" +str(RMSE_point_mass_angular_speed/RMS_angular_speed))
    txt.write("\n")
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()

def normalize_angle(theta):
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta = theta - 2 * math.pi
    return theta

def plot_first_integral(I_ground_truth, Y_imu, Y_mlp, Y_point_mass, pdf, txt):
    I_imu = np.zeros([I_ground_truth.shape[0], I_ground_truth.shape[1]])
    I_mlp = np.zeros([I_ground_truth.shape[0], I_ground_truth.shape[1]])
    I_point_mass = np.zeros([I_ground_truth.shape[0], I_ground_truth.shape[1]])
    I_imu [0,:] = I_ground_truth[0,:]
    I_mlp [0,:] = I_ground_truth[0,:]
    I_point_mass[0,:] = I_ground_truth[0,:]

    for k in range(I_ground_truth.shape[0]):
        if k>0:
            I_imu [k,:] = I_imu [k-1,:] + Y_imu [k,:] * 0.01 #by imu sensor 
            I_imu [k,1] = normalize_angle (I_imu [k,1])
            I_mlp [k,:] = I_mlp[k-1,:] + Y_mlp [k,:] * 0.01 #by MLP
            I_mlp [k,1] = normalize_angle (I_mlp [k,1])
            I_point_mass [k,:] = I_point_mass [k-1,:] + Y_point_mass [k,:] * 0.01 #by sim_point_mass
            I_point_mass [k,1] = normalize_angle (I_point_mass [k,1])

    RMSE_imu_speed = sqrt(mean_squared_error(I_imu[:,0],I_ground_truth[:,0]))
    RMSE_mlp_speed = sqrt(mean_squared_error(I_mlp[:,0],I_ground_truth[:,0]))
    RMSE_point_mass_speed = sqrt(mean_squared_error(I_point_mass[:,0],I_ground_truth[:,0]))
    RMS_speed = sqrt(sum(n*n for n in I_ground_truth[:,0])/len(I_ground_truth[:,0]))

    #plt.figure(figsize=(3,2))
    #plt.title("Speed")
    #plt.plot(I_ground_truth[:,0], color = 'blue',label = "Ground-truth speed")
    #plt.plot(I_imu[:,0], color = 'orange',label = "Incremental speed by sensors")
    #plt.plot(I_mlp[:,0], color = 'red', label = "Speed by MLP")
    #plt.plot(I_point_mass[:,0], color = 'green',label = "Speed by sim_point_mass")
    txt.write( "Speed RMSE of sensor:" + str(RMSE_imu_speed) + ", Error Rate:"+ str(RMSE_imu_speed/RMS_speed))
    txt.write("\n")
    txt.write( "Speed RMSE of MLP:" + str(RMSE_mlp_speed) + ", Error Rate:" + str(RMSE_mlp_speed/RMS_speed))
    txt.write("\n")
    txt.write( "Speed RMSE of sim_point_mass:" + str(RMSE_point_mass_speed) + ", Error Rate:" + str(RMSE_point_mass_speed/RMS_speed)) 
    txt.write("\n")
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()

    RMSE_imu_heading = sqrt(mean_squared_error(I_imu[:,1],I_ground_truth[:,1]))
    RMSE_mlp_heading = sqrt(mean_squared_error(I_mlp[:,1],I_ground_truth[:,1]))
    RMSE_point_mass_heading = sqrt(mean_squared_error(I_point_mass[:,1],I_ground_truth[:,1]))
    RMS_heading = sqrt(sum(n*n for n in I_ground_truth[:,1])/len(I_ground_truth[:,1]))

    #plt.figure(figsize=(3,2))
    #plt.title("Heading")
    #plt.plot(I_ground_truth[:,1], color = 'blue', label = "Ground-truth heading")
    #plt.plot(I_imu[:,1], color = 'orange', label = "Incremental heading by angular speed")
    #plt.plot(I_mlp[:,1], color = 'red', label = "Heading by MLP")
    #plt.plot(I_point_mass[:,1], color = 'green', label = "Heading by sim_point_mass")
    txt.write( "Heading RMSE of sensor:" + str(RMSE_imu_heading) + ", Error Rate:" + str(RMSE_imu_heading/RMS_heading))
    txt.write("\n")
    txt.write( "Heading RMSE of MLP:" + str(RMSE_mlp_heading) + ", Error Rate:" + str(RMSE_mlp_heading/RMS_heading))
    txt.write("\n")
    txt.write( "Heading RMSE of sim_point_mass:" + str(RMSE_point_mass_heading) + ", Error Rate:" +str(RMSE_point_mass_heading/RMS_heading))
    txt.write("\n")
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()
    return I_imu, I_mlp, I_point_mass

def plot_Trajectory(T_ground_truth, I_imu, I_mlp, I_point_mass, pdf, txt):
    T_imu = np.zeros([T_ground_truth.shape[0], T_ground_truth.shape[1]])
    T_mlp = np.zeros([T_ground_truth.shape[0], T_ground_truth.shape[1]])
    T_point_mass = np.zeros([T_ground_truth.shape[0], T_ground_truth.shape[1]])
    T_imu[0,:] = T_ground_truth[0,:]
    T_mlp[0,:] = T_ground_truth[0,:]
    T_point_mass[0,:] = T_ground_truth[0,:]
    Trajectory_length = 0

    for k in range(T_ground_truth.shape[0]):
        if k>0:
            T_imu[k,0] = T_imu[k-1,0] + I_imu[k,0]*np.cos(I_imu[k,1])*0.01
            T_imu[k,1] = T_imu[k-1,1] + I_imu[k,0]*np.sin(I_imu[k,1])*0.01
            T_mlp[k,0] = T_mlp[k-1,0] + I_mlp[k,0]*np.cos(I_mlp[k,1])*0.01
            T_mlp[k,1] = T_mlp[k-1,1] + I_mlp[k,0]*np.sin(I_mlp[k,1])*0.01
            T_point_mass[k,0] = T_point_mass[k-1,0] + I_point_mass[k,0]*np.cos(I_point_mass[k,1])*0.01
            T_point_mass[k,1] = T_point_mass[k-1,1] + I_point_mass[k,0]*np.sin(I_point_mass[k,1])*0.01
            Trajectory_length += np.sqrt((T_ground_truth[k,0] - T_ground_truth[k-1,0]) ** 2 + (T_ground_truth[k,1] - T_ground_truth[k-1,1]) ** 2)

    #fig = plt.figure(figsize=(3,2))
    #plt.title("Trajectory")
    #plt.plot(T_ground_truth[:,0], T_ground_truth[:,1], color = 'blue', label = "Ground-truth Tracjectory")
    #plt.plot(T_ground_truth[-1,0], T_ground_truth[-1,1], color = 'blue', marker = 'x')
    #plt.plot(T_imu[:,0], T_imu[:,1], color = 'orange', label = "Generated Tracjectory by IMU")
    #plt.plot(T_imu[-1,0], T_imu[-1,1], color = 'orange', marker = 'x')
    #plt.plot(T_mlp[:,0], T_mlp[:,1], color = 'red', label = "Tracjectory by MLP")
    #plt.plot(T_mlp[-1,0], T_mlp[-1,1],color = 'red', marker = 'x')
    #plt.plot(T_point_mass[:,0], T_point_mass[:,1], color = 'green',label = "Tracjectory by sim_point_mass")
    #plt.plot(T_point_mass[-1,0], T_point_mass[-1,1], color = 'green', marker = 'x')

    RMSE_imu_trajectory = sqrt(mean_squared_error(T_imu, T_ground_truth))
    RMSE_mlp_trajectory = sqrt(mean_squared_error(T_mlp, T_ground_truth))
    RMSE_point_mass_trajectory = sqrt(mean_squared_error(T_point_mass, T_ground_truth))
    txt.write( "Trajectory RMSE of sensor:" + str(RMSE_imu_trajectory) + ", Error Rate:" + str(RMSE_imu_trajectory/Trajectory_length))
    txt.write("\n")
    txt.write( "Trajectory RMSE of MLP:" + str(RMSE_mlp_trajectory) + ", Error Rate:" + str(RMSE_mlp_trajectory/Trajectory_length))
    txt.write("\n")
    txt.write( "Trajectory RMSE of sim_point_mass:" + str(RMSE_point_mass_trajectory) + ", Error Rate:" + str(RMSE_point_mass_trajectory/Trajectory_length)) 
    #pdf.savefig()  # saves the current figure into a pdf page
    #plt.close()

# TODO refactor model loading from binary file 
def load_model_refactor (filename):
    net_params = FnnModel()
    f = open(filename, "rb")
    net_params.ParseFromString(f.read())
    f.close()
    mean_param_norm = np.array(net_params.samples_mean)
    std_param_norm = np.array(net_params.samples_std)
    
    model = Sequential()
    model.add(Dense(net_params.layer[0].layer_output_dim,
                    input_dim=net_params.dim_input,
                    init='he_normal',
                    activation='relu',
                    weights=[np.array(net_params.layer[0].layer_input_weight),np.array(net_params.layer[0].layer_bias)]))
    for i in range(1, net_params.num_layer-1):
        model.add(Dense(net_params.layer[i].layer_output_dim,
                        init='he_normal',
                        activation='relu',
                         weights=[np.array(net_params.layer[i].layer_input_weight),np.array(net_params.layer[i].layer_bias)]))
    model.add(Dense(net_params.dim_output,
                    init='he_normal',
                    weights=[np.array(net_params.layer[num_layer-1].layer_input_weight),np.array(net_params.layer[num_layer-1].layer_bias)]))
    return model, mean_param_norm, std_param_norm


def evaluate(timestr, h5_segments, dirs = '/mnt/bos/modules/control/evaluation_result/'):               
#def evaluate(timestr, h5_segments, dirs = 'fueling/control/data/evaluation_result/'):#local dirs 

    X, I_ground_truth, Y_imu, Y_point_mass, T_ground_truth = generate_evaluation_data(h5_segments[1])

    #model = load_model ('fueling/control/data/model_output/fnn_model_weights_'+timestr+'.h5') #local dirs 
    #hf = h5py.File('fueling/control/data/model_output/fnn_model_norms_'+timestr+'.h5', 'r') #local dirs 
    model = load_model ('/mnt/bos/modules/control/dynamic_model_output/fnn_model_weights_'+timestr+'.h5') #bos dirs 
    hf = h5py.File('/mnt/bos/modules/control/dynamic_model_output/fnn_model_norms_'+timestr+'.h5', 'r') #bos dirs 
    mean_param_norm = np.array(hf.get('mean'))
    std_param_norm = np.array(hf.get('std'))
    hf.close()
    
    X = (X - mean_param_norm) / std_param_norm
    Y_mlp = model.predict(X)

    txt = open (dirs + 'evaluation_metrics_for_model_'+ timestr + '_under_scenario_' + h5_segments[0] + '.txt','w')
    with PdfPages(dirs + 'Trajectory_Visualization_' + timestr + '.pdf') as pdf:
        plot_model_output(Y_imu, Y_mlp, Y_point_mass, pdf, txt)
        I_imu, I_mlp, I_point_mass = plot_first_integral(I_ground_truth, Y_imu, Y_mlp, Y_point_mass, pdf, txt)
        plot_Trajectory(T_ground_truth, I_imu, I_mlp, I_point_mass, pdf, txt)
    txt.close()

