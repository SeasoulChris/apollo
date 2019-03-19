#!/usr/bin/env python

from datetime import datetime
from random import shuffle
from time import time
import os

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.regularizers import l1, l2
import h5py
import numpy as np

from fueling.control.features.parameters_training import dim
import fueling.common.colored_glog as glog
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
DIM_INPUT = dim["pose"] + dim["incremental"] + dim["control"]  # accounts for mps
DIM_OUTPUT = dim["incremental"]  # the speed mps is also output
INPUT_FEATURES = ["speed mps", "speed incremental",
                  "angular incremental", "throttle", "brake", "steering"]
DIM_LSTM_LENGTH = dim["timesteps"]
TIME_STEPS = 1
EPOCHS = 10


def setup_model(model_name):
    """
    set up neural network based on keras.Sequential
    model: output = relu(w2^T * tanh(w1^T * input + b1) + b2)
    """
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, activation='relu', W_regularizer=l2(0.001),
                   input_shape=(6, DIM_LSTM_LENGTH), init='he_normal'))
    if model_name == 'lstm_three_layer':
        model.add(Dense(4, init='he_normal', activation='relu'))
    model.add(Dense(2, init='he_normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def lstm_keras(lstm_input_data, lstm_output_data, param_norm, out_dir, model_name='lstm_two_layer'):
    for i in range(DIM_LSTM_LENGTH):
        lstm_input_data[:, :, i] = (lstm_input_data[:, :, i] - param_norm[0]) / param_norm[1]
    lstm_output_data[:, 0] = (lstm_output_data[:, 0] - param_norm[0][1]) / param_norm[1][1]
    lstm_output_data[:, 1] = (lstm_output_data[:, 1] - param_norm[0][2]) / param_norm[1][2]

    split_idx = int(lstm_input_data.shape[0] * 0.8 + 1)
    lstm_input_split = np.split(lstm_input_data, [split_idx])
    lstm_output_split = np.split(lstm_output_data, [split_idx])

    model = setup_model(model_name)
    training_history = model.fit(lstm_input_split[0], lstm_output_split[0],
                                 validation_data=(lstm_input_split[1], lstm_output_split[1]),
                                 epochs=EPOCHS, batch_size=64, verbose=2, shuffle=True)

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

   # save norm_params to hdf5
    h5_dir = os.path.join(out_dir, 'h5_model/lstm')
    model_dir = os.path.join(h5_dir, timestr)
    file_utils.makedirs(model_dir)
    norms_h5 = os.path.join(model_dir, 'norms.h5')
    with h5py.File(norms_h5, 'w') as h5_file:
        h5_file.create_dataset('mean', data=param_norm[0])
        h5_file.create_dataset('std', data=param_norm[1])
    weights_h5 = os.path.join(model_dir, 'weights.h5')
    model.save(weights_h5)
