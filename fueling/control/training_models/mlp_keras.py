#!/usr/bin/env python

from datetime import datetime
from random import shuffle
from time import time
import glob
import os
from keras.regularizers import l1, l2
from keras.layers import Dense, Input
from keras.layers import Activation
from keras.metrics import mse
from keras.models import Sequential, Model
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import google.protobuf.text_format as text_format
import h5py
import numpy as np

from fueling.control.features.parameters_training import dim
from modules.data.fuel.fueling.control.lib.proto.fnn_model_pb2 import FnnModel, Layer
import modules.data.fuel.fueling.control.lib.proto.fnn_model_pb2 as fnn_model_pb2

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
DIM_INPUT = dim["pose"] + dim["incremental"] + \
    dim["control"]  # accounts for mps
DIM_OUTPUT = dim["incremental"]  # the speed mps is also output
INPUT_FEATURES = ["speed mps", "speed incremental",
                  "angular incremental", "throttle", "brake", "steering"]
TIME_STEPS = 3


def setup_model(model_name):
    """
    set up neural network based on keras.Sequential
    model: output = relu(w2^T * tanh(w1^T * input + b1) + b2)
    """
    model = Sequential()
    model.add(Dense(10,
                    input_dim=5,
                    init='he_normal',
                    activation='relu',
                    W_regularizer=l2(0.001)))
    if model_name == 'mlp_three_layer':
        model.add(Dense(4,
                        init='he_normal',
                        activation='relu',
                        W_regularizer=l2(0.001)))
        print 'Load Three-layer MLP Model'
    model.add(Dense(2,
                    init='he_normal',
                    W_regularizer=l2(0.001)))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    return model


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as fin:
            for ds in fin.itervalues():
                if len(segments) == 0:
                    segments.append(np.array(ds))
                else:
                    segments[-1] = np.concatenate((segments[-1],
                                                   np.array(ds)), axis=0)
    # shuffle(segments)
    print('Segments count: ', len(segments))
    return segments


def generate_data(segments):
    total_len = 0
    TIME_STEPS = 3
    for segment in segments:
        total_len += (segment.shape[0] - TIME_STEPS)
    print "total_len = ", total_len
    x_data = np.zeros([total_len, DIM_INPUT])
    y_data = np.zeros([total_len, DIM_OUTPUT])
    # shuffle(segments)
    i = 0
    for segment in segments:
        for k in range(segment.shape[0]):
            if k >= TIME_STEPS:
                x_data[i, 0] = segment[k-TIME_STEPS, 14]  # speed mps
                x_data[i, 1] = segment[k-TIME_STEPS, 8] * \
                    np.cos(segment[k-TIME_STEPS, 0]) + segment[k-TIME_STEPS, 9] * \
                    np.sin(segment[k-TIME_STEPS, 0])  # acc
                x_data[i, 2] = segment[k-TIME_STEPS, 13]  # angular speed
                # control from chassis
                x_data[i, 3] = segment[k-TIME_STEPS, 15]
                # control from chassis
                x_data[i, 4] = segment[k-TIME_STEPS, 16]
                # control from chassis
                x_data[i, 5] = segment[k-TIME_STEPS, 17]
                y_data[i, 0] = segment[k, 8] * \
                    np.cos(segment[k, 0]) + segment[k, 9] * \
                    np.sin(segment[k, 0])  # acc next
                y_data[i, 1] = segment[k, 13]  # angular speed next
                i += 1
    # window size 51, polynomial order 3
    x_data[:, 1] = savgol_filter(x_data[:, 1], 51, 3)
    # window size 51, polynomial order 3
    y_data[:, 0] = savgol_filter(y_data[:, 0], 51, 3)
    return x_data, y_data


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
    net_params.dim_input = DIM_INPUT
    net_params.dim_output = DIM_OUTPUT
    with open(filename, 'wb') as params_file:
        params_file.write(net_params.SerializeToString())
    # print text_format.MessageToString(net_params)


def mlp_keras(hdf5, out_dirs, model_name='mlp_two_layer'):
    segments = generate_segments(hdf5)
    x_data, y_data = generate_data(segments)
    param_norm = get_param_norm(x_data)
    x_data = (x_data - param_norm[0]) / param_norm[1]
    x_data = x_data[:, [0, 1, 3, 4, 5]]
    y_data[:, 0] = (y_data[:, 0] - param_norm[0][1]) / param_norm[1][1]
    y_data[:, 1] = (y_data[:, 1] - param_norm[0][2]) / param_norm[1][2]
    print "x shape = ", x_data.shape
    print "y shape = ", y_data.shape

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)
    print "x_train shape = ", x_train.shape
    print "y_train shape = ", y_train.shape

    model = setup_model(model_name)
    training_history = model.fit(x_train, y_train,
                                 shuffle=True,
                                 nb_epoch=30,
                                 batch_size=32)

    evaluation = model.evaluate(x_test, y_test)
    print "\nModel evaluation: "
    print "Loss on testing data is ", evaluation[0]
    print "MSE on testing data is ", evaluation[1]

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model(model, param_norm, out_dirs + 'mlp_model_' + timestr + '.bin')

    # save norm_params to hdf5
    h5_file = h5py.File(out_dirs + 'mlp_model_norms_' + timestr + '.h5', 'w')
    h5_file.create_dataset('mean', data=param_norm[0])
    h5_file.create_dataset('std', data=param_norm[1])
    h5_file.close()

    model.save(out_dirs + 'mlp_model_weights_' + timestr + '.h5')
