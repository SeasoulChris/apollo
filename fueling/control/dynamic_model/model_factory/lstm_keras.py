#!/usr/bin/env python

from datetime import datetime
from random import shuffle
from time import time
import os

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.regularizers import l1, l2
import colored_glog as glog
import h5py
import numpy as np

from fueling.control.dynamic_model.conf.model_config import feature_config, lstm_model_config
import fueling.common.file_utils as file_utils


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
IS_HOLISTIC = feature_config["is_holistic"]
IS_BACKWARD = feature_config["is_backward"]
DIM_INPUT = feature_config["holistic_input_dim"] if IS_HOLISTIC else feature_config["input_dim"]
DIM_OUTPUT = feature_config["holistic_output_dim"] if IS_HOLISTIC else feature_config["output_dim"]
DIM_LSTM_LENGTH = feature_config["sequence_length"]
EPOCHS = lstm_model_config["epochs"]


def setup_model(model_name):
    """
    set up neural network based on keras.Sequential
    model: output = relu(w2^T * tanh(w1^T * input + b1) + b2)
    """
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, activation='relu', W_regularizer=l2(0.001),
                   input_shape=(DIM_INPUT, DIM_LSTM_LENGTH), init='he_normal'))
    if model_name == 'lstm_three_layer':
        model.add(Dense(4, init='he_normal', activation='relu'))
    model.add(Dense(DIM_OUTPUT, init='he_normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def lstm_keras(lstm_input_data, lstm_output_data, param_norm, out_dir, model_name='lstm_two_layer'):
    glog.info("Start to train LSTM model")
    (input_fea_mean, input_fea_std), (output_fea_mean, output_fea_std) = param_norm
    for i in range(DIM_LSTM_LENGTH):
        lstm_input_data[:, :, i] = (lstm_input_data[:, :, i] - input_fea_mean) / input_fea_std
    lstm_output_data = (lstm_output_data - output_fea_mean) / output_fea_std

    split_idx = int(lstm_input_data.shape[0] * 0.8 + 1)
    lstm_input_split = np.split(lstm_input_data, [split_idx])
    lstm_output_split = np.split(lstm_output_data, [split_idx])

    model = setup_model(model_name)
    training_history = model.fit(lstm_input_split[0], lstm_output_split[0],
                                 validation_data=(lstm_input_split[1], lstm_output_split[1]),
                                 epochs=EPOCHS, batch_size=64, verbose=2, shuffle=True)

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

   # save norm_params and model_weights to hdf5
   # save norm_params and model_weights to hdf5
    if IS_BACKWARD:
        h5_model_dir = os.path.join(out_dir, 'h5_model/lstm/backward')
    else:
        h5_model_dir = os.path.join(out_dir, 'h5_model/lstm/forward')
        
    h5_file_dir = os.path.join(h5_model_dir, timestr)
    file_utils.makedirs(h5_file_dir)

    norms_h5 = os.path.join(h5_file_dir, 'norms.h5')
    with h5py.File(norms_h5, 'w') as h5_file:
        h5_file.create_dataset('input_mean', data=input_fea_mean)
        h5_file.create_dataset('input_std', data=input_fea_std)
        h5_file.create_dataset('output_mean', data=output_fea_mean)
        h5_file.create_dataset('output_std', data=output_fea_std)

    weights_h5 = os.path.join(h5_file_dir, 'weights.h5')
    model.save(weights_h5)
