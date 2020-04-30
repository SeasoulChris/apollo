#!/usr/bin/env python

"""Extracting and processing dataset"""
import argparse
import glob
import os
import pickle
import sys

# disable GPU for local test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import load_model
import h5py
import numpy as np

from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config, input_index, output_index
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

# Cache models to avoid the same one got loaded repeatedly


class GlobalModels(object):
    """The global pool for models"""

    models = {}

    @staticmethod
    def add_model(model_file_path, model):
        """Add a new model to pool"""
        GlobalModels.models[model_file_path] = model

    @staticmethod
    def get_model(model_file_path):
        """Get a model from pool"""
        if not model_file_path in GlobalModels.models:
            return None
        return GlobalModels.models[model_file_path]


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


def generate_gp_data(model_path, segment):
    """
    Generate one sample (input_segment, output_segment) from a segment
    """
    model_norms_path = os.path.join(model_path, 'norms.h5')
    with h5py.File(model_norms_path, 'r') as model_norms_file:
        input_mean = np.array(model_norms_file.get('input_mean'))
        input_std = np.array(model_norms_file.get('input_std'))
        output_mean = np.array(model_norms_file.get('output_mean'))
        output_std = np.array(model_norms_file.get('output_std'))
        norms = (input_mean, input_std, output_mean, output_std)

    model_weights_path = os.path.join(model_path, 'weights.h5')

    logging.info(F'loading model: {model_weights_path}')
    model = GlobalModels.get_model(model_weights_path)
    if not model:
        model = load_model(model_weights_path)
        GlobalModels.add_model(model_weights_path, model)

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

        predicted_a, predicted_w = generate_mlp_output(input_segment[k, 0: MLP_DIM_INPUT].reshape(
                                                       1, MLP_DIM_INPUT), model, norms)
        # Calculate the model prediction on current speed and heading
        prev_v = predicted_v  # previous speed
        predicted_v += predicted_a * feature_config["delta_t"]  # updated speed v = v0 + acc * dt
        predicted_heading += predicted_w * feature_config["delta_t"]

        # Calculate the model prediction on current position
        ds = prev_v * feature_config["delta_t"] + 0.5 * predicted_a * \
            feature_config["delta_t"] * feature_config["delta_t"]
        # ds = _distance_s(predicted_a, feature_config["delta_t"], prev_v)
        predicted_x += ds * np.cos(predicted_heading)
        predicted_y += ds * np.sin(predicted_heading)

    # def _distance_s(acc, dt, v0):
    #     return v0 * dt + 0.5 * acc * dt * dt
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
    # logging.info(
    # f'input mean: {input_mean}, input_std: {input_std}, output_mean:{output_mean}, output_std:{output_std}')
    # Prediction on acceleration and angular speed by MLP
    output_fnn = np.zeros([1, 2])
    cur_speed = mlp_input[0, 0]
    # Normalization for MLP model's input/output
    mlp_input[0, :] = (mlp_input[0, :] - input_mean) / input_std
    output_fnn[0, :] = model.predict(mlp_input)
    output_fnn[0, :] = output_fnn[0, :] * output_std + output_mean
    # Update the vehicle speed based on predicted acceleration
    velocity_fnn = output_fnn[0, 0] * feature_config["delta_t"] + cur_speed
    # logging.info(velocity_fnn)
    # If (negative speed under forward gear || positive speed under backward gear ||
    #     neutral gear):
    # Then truncate speed, acceleration, and angular speed to 0
    if gear_status * (velocity_fnn + gear_status * SPEED_EPSILON) <= 0:
        logging.info(f'output is set as zeros')
        output_fnn[0, 0] = 0.0
        output_fnn[0, 1] = 0.0
    logging.debug(f'mlp input is {mlp_input}')
    logging.debug(f'mlp output is {output_fnn[0,0]} and {output_fnn[0,1]}')

    return output_fnn[0, 0], output_fnn[0, 1]


def get_train_data(args):
    """
    Generate labeled data from a list of hdf5 files (unlabled data)
    """
    datasets = glob.glob(os.path.join(args.unlabeled_dataset_path, '*.hdf5'))
    file_utils.makedirs(args.labeled_dataset_path)
    path_suffix = ".hdf5"

    for h5_file in datasets:
        pre_file_name = h5_file.split(args.unlabeled_dataset_path)[1].split(path_suffix)[0]
        file_name = os.path.join(args.labeled_dataset_path, pre_file_name + '.h5')
        if os.path.exists(file_name):
            logging.info("File Already Generated: {}".format(file_name))
            continue
        # generated data segment for unhandled file
        segment = generate_segment(h5_file)
        input_segment, output_segment = generate_gp_data(args.model_path, segment)
        # save the generated label dataset
        with h5py.File(file_name, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)


if __name__ == '__main__':
    logging.info("running....")
    parser = argparse.ArgumentParser(description='Label')
    parser.add_argument('--unlabeled_dataset_path', type=str,
                        default="./apollo/data/mbg_1_all/")
    # parser.add_argument('--unlabeled_dataset_path', type=str,
    # default="/fuel/fueling/control/dynamic_model_2_0/testdata/train_data/")
    parser.add_argument('--model_path', type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/mlp_model/forward/")
    # parser.add_argument('--labeled_dataset_path', type=str,
    #                     default="/fuel/fueling/control/dynamic_model_2_0/testdata/labeled_data/")
    parser.add_argument('--labeled_dataset_path', type=str,
                        default="./apollo/data/labeled_data")
    args = parser.parse_args()
    get_train_data(args)
