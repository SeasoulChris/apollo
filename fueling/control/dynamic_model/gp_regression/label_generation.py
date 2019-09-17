#!/usr/bin/env python

"""Extracting and processing dataset"""
import argparse
import glob
import os
import pickle
import sys

from keras.models import load_model
import h5py
import numpy as np

from fueling.control.dynamic_model.gp_regression.model_conf import segment_index, feature_config
from fueling.control.dynamic_model.gp_regression.model_conf import input_index, output_index
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
# INPUT_LENGTH = feature_config["DELTA_T"] / feature_config["delta_t"]
INPUT_LENGTH = 100
DIM_INPUT = feature_config["input_dim"]
MLP_DIM_INPUT = feature_config["mlp_input_dim"]
DIM_OUTPUT = feature_config["output_dim"]
SPEED_EPSILON = 1e-6   # Speed Threshold To Indicate Driving Directions
PI = 3.14159


def generate_segment(h5_file):
    """
    load a single h5 file to a numpy array
    """
    segment = None
    logging.info('Loading New File {}'.format(h5_file))
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


def generate_gp_data(args, segment):
    """
    Generate one sample (input_segment, output_segment) from a segment
    """
    model_norms_path = os.path.join(args.model_path, 'norms.h5')
    with h5py.File(model_norms_path, 'r') as model_norms_file:
        input_mean = np.array(model_norms_file.get('input_mean'))
        input_std = np.array(model_norms_file.get('input_std'))
        output_mean = np.array(model_norms_file.get('output_mean'))
        output_std = np.array(model_norms_file.get('output_std'))
        norms = (input_mean, input_std, output_mean, output_std)
    model_weights_path = os.path.join(args.model_path, 'weights.h5')
    model = load_model(model_weights_path)

    input_segment = np.zeros([INPUT_LENGTH, DIM_INPUT])
    output_segment = np.zeros([DIM_OUTPUT])
    # Initialize the first frame's data
    predicted_v = segment[0, segment_index["speed"]]
    predicted_heading = segment[0, segment_index["heading"]]
    predicted_w = segment[0, segment_index["w_z"]]
    predicted_x = segment[0, segment_index["x"]]
    predicted_y = segment[0, segment_index["y"]]

    for k in range(INPUT_LENGTH):
        input_segment[k, input_index["v"]] = segment[k, segment_index["speed"]]
        input_segment[k, input_index["a"]] = segment[k, segment_index["a_x"]] * \
            np.cos(segment[k, segment_index["heading"]]) + \
            segment[k, segment_index["a_y"]] * \
            np.sin(segment[k, segment_index["heading"]])
        input_segment[k, input_index["u_1"]] = segment[k, segment_index["throttle"]]
        input_segment[k, input_index["u_2"]] = segment[k, segment_index["brake"]]
        input_segment[k, input_index["u_3"]] = segment[k, segment_index["steering"]]
        input_segment[k, input_index["phi"]] = segment[k, segment_index["heading"]] / PI

        # TODO(Jiaxuan): Solve the keras error and get the MLP model's (x,y) prediction
        predicted_a, predicted_w = generate_mlp_output(input_segment[k, 0 : MLP_DIM_INPUT].reshape(
                                                       1, MLP_DIM_INPUT), model, norms)
        # Calculate the model prediction on current speed and heading
        predicted_v += predicted_a * feature_config["delta_t"]
        predicted_heading += predicted_w * feature_config["delta_t"]
        # Calculate the model prediction on current position
        predicted_x += predicted_v * np.cos(predicted_heading) * feature_config["delta_t"]
        predicted_y += predicted_v * np.sin(predicted_heading) * feature_config["delta_t"]
    # logging.info("The predicted x:{}, y:{}".format(predicted_x, predicted_y))
    # The residual error on x and y prediction
    output_segment[output_index["d_x"]] = segment[INPUT_LENGTH -
                                                  1, segment_index["x"]] - predicted_x
    output_segment[output_index["d_y"]] = segment[INPUT_LENGTH -
                                                  1, segment_index["y"]] - predicted_y
    logging.info("Residual Error x:{}, y:{}".format(output_segment[0], output_segment[1]))
    return (input_segment, output_segment)


def generate_mlp_output(mlp_input, model, norms, gear_status=1):
    """
    Generate MLP model's direct output
    """
    input_mean, input_std, output_mean, output_std = norms
    # Prediction on acceleration and angular speed by MLP
    output_fnn = np.zeros([1, 2])
    # Normalization for MLP model's input/output
    mlp_input[0, :] = (mlp_input[0, :] - input_mean) / input_std
    # logging.info("Model Input {}".format(mlp_input))
    output_fnn[0, :] = model.predict(mlp_input)
    output_fnn[0, :] = output_fnn[0, :] * output_std + output_mean
    # logging.info("Model Output {}".format(output_fnn))
    # Update the vehicle speed based on predicted acceleration
    velocity_fnn = output_fnn[0, 0] * feature_config["delta_t"] + mlp_input[0, 0]
    # If (negative speed under forward gear || positive speed under backward gear ||
    #     neutral gear):
    # Then truncate speed, acceleration, and angular speed to 0
    if gear_status * (velocity_fnn + gear_status * SPEED_EPSILON) <= 0:
        output_fnn[0, 0] = 0.0
        output_fnn[0, 1] = 0.0

    return output_fnn[0, 0], output_fnn[0, 1]


def get_train_data(args):
    """
    Generate labeled data from a list of h5 files (unlabled data)
    """
    datasets = glob.glob(os.path.join(args.unlabeled_dataset_path, '*.hdf5'))
    file_utils.makedirs(args.labeled_dataset_path)
    path_suffix = ".hdf5"

    for h5_file in datasets:
        file_name = h5_file.split(args.unlabeled_dataset_path)[1].split(path_suffix)[0]
        file_name = os.path.join(args.labeled_dataset_path, file_name + '.h5')
        if os.path.exists(file_name):
            logging.info("File Already Generated: {}".format(file_name))
            continue
        # generated data segment for unhandled file
        segment = generate_segment(h5_file)
        input_segment, output_segment = generate_gp_data(args, segment)
        # save the generated label dataset
        with h5py.File(file_name, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)


if __name__ == '__main__':
    """
    Temporarily run in py27 environment
    """
    parser = argparse.ArgumentParser(description='Label')
    # paths
    parser.add_argument('--unlabeled_dataset_path', type=str,
                        default="testdata/control/gaussian_process/dataset/unlabeled_dataset/bigloop_1/")
    parser.add_argument('--model_path', type=str,
                        default="testdata/control/gaussian_process/mlp_model/forward/")
    parser.add_argument('--labeled_dataset_path', type=str,
                        default="testdata/control/gaussian_process/dataset/label_generation/")
    args = parser.parse_args()
    get_train_data(args)
