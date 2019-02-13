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

import fueling.control.lib.proto.fnn_model_pb2 as fnn_model_pb2
from fueling.control.lib.proto.fnn_model_pb2 import FnnModel, Layer
from fueling.control.features.parameters_training import dim
from time import time
import os
import glob
import h5py
import numpy as np
import sys
from datetime import datetime
import google.protobuf.text_format as text_format
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.metrics import mse
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils
from keras.regularizers import l2, l1
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from random import choice
from random import randint
from random import shuffle


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
dim_input = dim["pose"] + dim["incremental"] + \
    dim["control"]  # accounts for mps
dim_output = dim["incremental"]  # the speed mps is also output
input_features = ["speed mps", "speed incremental",
                  "angular incremental", "throttle", "brake", "steering"]


def setup_model(model_name):
    if model_name == 'mlp_three_layer':
        with open('fueling/control/conf/mlp_three_layer.json', 'r') as f:
            model = model_from_json(f.read())
        print ('Load Three-layer MLP Model')
    else:
        with open('fueling/control/conf/mlp_two_layer.json', 'r') as f:
            model = model_from_json(f.read())
        print ('Load Two-layer MLP Model')
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    return model


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as f:
            names = [n for n in f.keys()]
            if len(names) < 1:
                continue
            for i in range(len(names)):
                ds = np.array(f[names[i]])
                segments.append(ds)
    shuffle(segments)
    print('Segments count: ', len(segments))
    return segments


def generate_data(segments):
    total_len = 0
    for i in range(len(segments)):
        total_len += (segments[i].shape[0] - 2)
    print "total_len = ", total_len
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    print "Y size = ", Y.shape
    shuffle(segments)
    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(segment.shape[0] - 1):
            if k > 0:
                X[i, 0] = segment[k-1, 14]  # speed mps
                X[i, 1] = segment[k-1, 8] * \
                    np.cos(segment[k-1, 0]) + segment[k-1, 9] * \
                    np.sin(segment[k-1, 0])  # acc
                X[i, 3] = segment[k-1, 15]  # control from chassis
                X[i, 4] = segment[k-1, 16]  # control from chassis
                X[i, 5] = segment[k-1, 17]  # control from chassis
                Y[i, 0] = segment[k, 8] * \
                    np.cos(segment[k, 0]) + segment[k, 9] * \
                    np.sin(segment[k, 0])  # acc next
                Y[i, 1] = segment[k, 13]  # angular speed next
                i += 1
    # window size 51, polynomial order 3
    X[:, 1] = savgol_filter(X[:, 1], 51, 3)
    # window size 51, polynomial order 3
    Y[:, 0] = savgol_filter(Y[:, 0], 51, 3)
    #hf = h5py.File('training_data.h5', 'w')
    #hf.create_dataset('training_X', data=X)
    #hf.create_dataset('training_Y', data=Y)
    return X, Y


def get_param_norm(feature):
    """
    normalize the samples and save normalized parameters
    """
    fea_mean = np.mean(feature, axis=0)
    print "feature mean = ", fea_mean
    fea_std = np.std(feature, axis=0) + 1e-6
    print "feature std = ", fea_std
    param_norm = (fea_mean, fea_std)
    return param_norm


def save_model(model, param_norm, filename):
    """
    save the trained model parameters into protobuf binary format file
    """
    net_params = FnnModel()
    net_params.samples_mean.columns.extend(param_norm[0].reshape(-1).tolist())
    net_params.samples_std.columns.extend(param_norm[1].reshape(-1).tolist())
    net_params.num_layer = 0
    previous_dim = 0
    for layer in model.layers:
        net_params.num_layer += 1
        net_layer = net_params.layer.add()
        config = layer.get_config()
        print config
        if net_params.num_layer == 1:
            net_layer.layer_input_dim = config['batch_input_shape'][1]
            net_layer.layer_output_dim = config['units']
            previous_dim = net_layer.layer_output_dim
        else:
            net_layer.layer_input_dim = previous_dim
            net_layer.layer_output_dim = config['units']
            previous_dim = net_layer.layer_output_dim

        if config['activation'] == 'relu':
            net_layer.layer_activation_func = fnn_model_pb2.Layer.RELU
        elif config['activation'] == 'tanh':
            net_layer.layer_activation_func = fnn_model_pb2.Layer.TANH
        elif config['activation'] == 'sigmoid':
            net_layer.layer_activation_func = fnn_model_pb2.Layer.SIGMOID

        weights, bias = layer.get_weights()
        net_layer.layer_bias.columns.extend(bias.reshape(-1).tolist())
        for col in weights.tolist():
            row = net_layer.layer_input_weight.rows.add()
            row.columns.extend(col)
    net_params.dim_input = dim_input
    net_params.dim_output = dim_output
    with open(filename, 'wb') as params_file:
        params_file.write(net_params.SerializeToString())
    # print text_format.MessageToString(net_params)


def mlp_keras(model_name = 'mlp_two_layer', out_dirs = '/mnt/bos/modules/control/dynamic_model_output/'):

    # NOTE: YOU MAY NEED TO CHANGE THIS PATH ACCORDING TO YOUR ENVIRONMENT
    hdf5 = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_training/*.hdf5')
    print "hdf5 files are :"
    print hdf5

    segments = generate_segments(hdf5)
    X, Y = generate_data(segments)

    print "X shape = ", X.shape
    print "Y shape = ", Y.shape

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    print "X_train shape = ", X_train.shape
    print "Y_train shape = ", Y_train.shape

    param_norm = get_param_norm(X_train)
    X_train = (X_train - param_norm[0]) / param_norm[1]

    model = setup_model(model_name)
    model.fit(X_train, Y_train,
              shuffle=True,
              nb_epoch=30,
              batch_size=32)

    X_test = (X_test - param_norm[0]) / param_norm[1]
    # plot_H5_features_hist(X_train)

    evaluation = model.evaluate(X_test, Y_test)
    print "\nModel evaluation: "
    print "Loss on testing data is ", evaluation[0]
    print "MSE on testing data is ", evaluation[1]

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model(model, param_norm, out_dirs + 'fnn_model_' + timestr + '.bin')
    
    # save norm_params to hdf5
    hf = h5py.File(out_dirs + 'fnn_model_norms_' + timestr + '.h5', 'w')
    hf.create_dataset('mean', data = param_norm[0])
    hf.create_dataset('std', data = param_norm[1])
    hf.close()

    model.save(out_dirs + 'fnn_model_weights_' + timestr + '.h5')

