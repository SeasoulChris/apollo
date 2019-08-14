#!/usr/bin/env python

"""Extracting and processing dataset"""
import glob
import os
import pickle
import sys

from keras.models import load_model
from liegroups.torch import SO3
from torch.utils.data.dataset import Dataset
import colored_glog as glog
import h5py
import numpy as np
import torch

from fueling.control.dynamic_model.gp_regression.model_conf import segment_index, feature_config
from fueling.control.dynamic_model.gp_regression.model_conf import input_index, output_index

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
# INPUT_LENGTH = feature_config["DELTA_T"] / feature_config["delta_t"]
INPUT_LENGTH = 100
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]
SPEED_EPSILON = 1e-6   # Speed Threshold To Indicate Driving Directions


class GPDataSet(Dataset):

    def __init__(self, args):
        """Initialization"""
        self.data_path = args.data_path

    def generate_segment(self, h5_file):
        """
        load a single h5 file to a numpy array
        """
        segment = None
        glog.info('Loading {}'.format(h5_file))
        with h5py.File(h5_file, 'r') as fin:
            for ds in fin.values():
                if segment is None:
                    segment = np.array(ds)
                else:
                    segment = np.concatenate((segment, np.array(ds)), axis=0)
        if segment.shape[0] != INPUT_LENGTH:
            raise Exception('File {} has illegal number of frames: {}'.format(h5_file,
                                                                              segment.shape[0]))
        return segment

    def generate_gp_data(self, segment):
        """
        Generate one sample (input_segment, output_segment) from a segment
        """
        input_segment = torch.zeros(1, INPUT_LENGTH, DIM_INPUT)
        output_segment = torch.zeros(DIM_OUTPUT, 1)
        # Initialize the first frame's data
        predicted_speed = segment[0, segment_index["speed"]]
        predicted_a = segment[0, segment_index["a_x"]] * \
            np.cos(segment[0, segment_index["heading"]]) + \
            segment[0, segment_index["a_y"]] * \
            np.sin(segment[0, segment_index["heading"]])
        predicted_heading = segment[0, segment_index["heading"]]
        predicted_w = segment[0, segment_index["w_z"]]

        for k in range(INPUT_LENGTH):
            input_segment[0, k, input_index["v"]] = segment[k, segment_index["speed"]]
            input_segment[0, k, input_index["a"]] = segment[k, segment_index["a_x"]] * \
                np.cos(segment[k, segment_index["heading"]]) + \
                segment[k, segment_index["a_y"]] * \
                np.sin(segment[k, segment_index["heading"]])
            input_segment[0, k, input_index["u_1"]] = segment[k, segment_index["throttle"]]
            input_segment[0, k, input_index["u_2"]] = segment[k, segment_index["brake"]]
            input_segment[0, k, input_index["u_3"]] = segment[k, segment_index["steering"]]

            # TODO(Jiaxuan): Solve the keras error and get the MLP model's (x,y) prediction
            # model_predicted_a, model_predicted_w = generate_mlp_output(gp_input_data[k , :],
            #                                                            mlp_model_folder)
            # Calculate the model prediction on current speed and heading
            # model_predicted_speed += model_predicted_a * feature_config["delta_t"]
            # model_predicted_heading += model_predicted_w * feature_config["delta_t"]
            # Calculate the model prediction on current position
            # model_predicted_x += model_predicted_speed * np.cos(model_predicted_heading) * \
            #                      feature_config["delta_t"]
            # model_predicted_y += model_predicted_speed * np.sin(model_predicted_heading) * \
            #                      feature_config["delta_t"]

        # The residual error on x and y prediction
        output_segment[output_index["d_x"], 0] = segment[INPUT_LENGTH - 1, segment_index["x"]] - \
            segment[0, segment_index["x"]]
        output_segment[output_index["d_y"], 0] = segment[INPUT_LENGTH - 1, segment_index["y"]] - \
            segment[0, segment_index["y"]]
        return (input_segment, output_segment)

    def generate_mlp_output(mlp_input, model_folder, gear_status=1):
        """
        Generate MLP model's direct output
        """
        model_norms_path = os.path.join(model_folder, 'norms.h5')
        with h5py.File(model_norms_path, 'r') as model_norms_file:
            input_mean = np.array(model_norms_file.get('input_mean'))
            input_std = np.array(model_norms_file.get('input_std'))
            output_mean = np.array(model_norms_file.get('output_mean'))
            output_std = np.array(model_norms_file.get('output_std'))

        model_weights_path = os.path.join(model_folder, 'weights.h5')
        model = load_model(model_weights_path)
        # Prediction on acceleration and angular speed by MLP
        output_fnn = np.zeros([1, 2])

        # Normalization for MLP model's input/output
        mlp_input[0, :] = (mlp_input[0, :] - input_mean) / input_std
        output_fnn[0, :] = model.predict(mlp_input)
        output_fnn[0, :] = output_fnn[0, :] * output_std + output_mean

        # Update the vehicle speed based on predicted acceleration
        velocity_fnn = output_fnn[0, 0] * feature_config["delta_t"] + mlp_input[0, 0]
        # If (negative speed under forward gear || positive speed under backward gear ||
        #     neutral gear):
        # Then truncate speed, acceleration, and angular speed to 0
        if gear_status * (velocity_fnn + gear_status * SPEED_EPSILON) <= 0:
            output_fnn[0, 0] = 0.0
            output_fnn[0, 1] = 0.0

        return output_fnn[0, 0], output_fnn[0, 1]

    def get_train_data(self):
        """
        Generate training data from a list of h5 files
        """
        datasets = glob.glob(os.path.join(self.data_path, '*.hdf5'))
        input_data = torch.zeros(0, INPUT_LENGTH, DIM_INPUT)
        output_data = torch.zeros(DIM_OUTPUT, 0)
        for h5_file in datasets:
            segment = self.generate_segment(h5_file)
            input_segment, output_segment = self.generate_gp_data(segment)
            input_data = torch.cat((input_data, input_segment), 0)
            output_data = torch.cat((output_data, output_segment), 1)
        return (input_data, output_data)
