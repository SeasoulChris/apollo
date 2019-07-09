#!/usr/bin/env python

import os
import sys

from scipy.signal import savgol_filter
import colored_glog as glog
import h5py
import numpy as np

from fueling.control.dynamic_model.conf.model_config import acc_method
from fueling.control.dynamic_model.conf.model_config import feature_config
from fueling.control.dynamic_model.conf.model_config import segment_index


# Constants
DIM_SEQUENCE_LENGTH = feature_config["sequence_length"]
DIM_DELAY_STEPS = feature_config["delay_steps"]
MAXIMUM_SEGMENT_LENGTH = feature_config["maximum_segment_length"]
STD_EPSILON = 1e-6
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]


def generate_segment(h5_file):
    """
    load a single h5 file to a numpy array
    """
    segment = None
    glog.info('Loading {}'.format(h5_file))
    with h5py.File(h5_file, 'r') as fin:
        for ds in fin.itervalues():
            if segment is None:
                segment = np.array(ds)
            else:
                segment = np.concatenate((segment, np.array(ds)), axis=0)
    glog.info('The length of frames {}'.format(segment.shape[0]))
    return segment


def generate_segment_from_list(hdf5_file_list):
    """
    load a list of h5 files to a numpy array
    """
    segment = None
    for filename in hdf5_file_list:
        glog.info('Processing file %s' % filename)
        with h5py.File(filename, 'r') as fin:
            for value in fin.values():
                if segment is None:
                    segment = np.array(value)
                else:
                    segment = np.concatenate((segment, np.array(value)), axis=0)
    return segment


def differential(first_elem, second_elem, dt):
    """
    detrivative
    """
    return (second_elem - first_elem) / dt


def vect_differential(vect, dt):
    """
    calculate detrivative of a vector
    """
    ret = np.zeros(vect.shape)
    print("vect shape", vect.shape)
    print("vect len", len(vect))
    for index in range(1, len(vect)):
        ret[index - 1] = (vect[index] - vect[index - 1]) / dt
    ret[index] = ret[index - 1]
    return ret


def vect_ddiff(vect, dt):
    """
    calculate secondary detrivative of a vector
    """
    pre_ret = vect_differential(vect, dt)
    ret = vect_differential(vect, dt)
    return ret


def feature_preprocessing(segment):
    """
    smooth noisy raw data from IMU by savgol_filter
    """
    # discard the segments that are too short
    if segment.shape[0] < WINDOW_SIZE or segment.shape[0] < DIM_DELAY_STEPS + DIM_SEQUENCE_LENGTH:
        return None
    # discard the segments that are too long
    if segment.shape[0] > MAXIMUM_SEGMENT_LENGTH:
        return None
    # correct the localization outliers
    outlier_data_correction(segment)

    # smooth localization position data
    if not acc_method["acc_from_IMU"]:
        segment[:, segment_index["x"]] = savgol_filter(
            segment[:, segment_index["x"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
        segment[:, segment_index["y"]] = savgol_filter(
            segment[:, segment_index["y"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
        # speed
        tmp_x_v = vect_differential(
            segment[:, segment_index["x"]], feature_config["delta_t"])
        tmp_x_v = savgol_filter(
            tmp_x_v, WINDOW_SIZE, POLYNOMINAL_ORDER)
        tmp_y_v = vect_differential(
            segment[:, segment_index["y"]], feature_config["delta_t"])
        tmp_y_v = savgol_filter(
            tmp_y_v, WINDOW_SIZE, POLYNOMINAL_ORDER)
        # acc
        segment[:, segment_index["a_x"]] = vect_differential(
            tmp_x_v, feature_config["delta_t"])
        segment[:, segment_index["a_x"]] = savgol_filter(savgol_filter(
            segment[:, segment_index["a_x"]], WINDOW_SIZE, POLYNOMINAL_ORDER), WINDOW_SIZE, POLYNOMINAL_ORDER)
        segment[:, segment_index["a_y"]] = vect_differential(
            tmp_y_v, feature_config["delta_t"])
        segment[:, segment_index["a_y"]] = savgol_filter(savgol_filter(
            segment[:, segment_index["a_y"]], WINDOW_SIZE, POLYNOMINAL_ORDER), WINDOW_SIZE, POLYNOMINAL_ORDER)
    else:
        # smooth IMU acceleration data
        segment[:, segment_index["a_x"]] = savgol_filter(
            segment[:, segment_index["a_x"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
        segment[:, segment_index["a_y"]] = savgol_filter(
            segment[:, segment_index["a_y"]], WINDOW_SIZE, POLYNOMINAL_ORDER)

    segment[:, segment_index["w_z"]] = savgol_filter(
        segment[:, segment_index["w_z"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
    return segment


def outlier_data_correction(segment):
    """
    correct the localization error regarding the acceleration and angular velocity values
    some values are doubled abnormally and periodically every 3 or 4 frames
    """
    # iterate the segment and check 3 points each time
    # if the middle point is abnormally large, divide it by 2
    for k in range(segment.shape[0] - 2):
        if abs(segment[k + 1, segment_index["a_x"]]) > 0.75 * (
                abs(segment[k, segment_index["a_x"]]) + abs(segment[k + 2, segment_index["a_x"]])):
            segment[k + 1, segment_index["a_x"]] /= 2.0
        if abs(segment[k + 1, segment_index["a_y"]]) > 0.75 * (
                abs(segment[k, segment_index["a_y"]]) + abs(segment[k + 2, segment_index["a_y"]])):
            segment[k + 1, segment_index["a_y"]] /= 2.0
        if abs(segment[k + 1, segment_index["w_z"]]) > 0.75 * (
                abs(segment[k, segment_index["w_z"]]) + abs(segment[k + 2, segment_index["w_z"]])):
            segment[k + 1, segment_index["w_z"]] /= 2.0
    # delete the first and last column since they are not checked
    segment = segment[:, 1:-1]
    return segment


def get_param_norm(input_feature, output_feature):
    """
    normalize the samples and save normalized parameters
    """
    glog.info("Input Feature Dimension {}".format(input_feature.shape))
    glog.info("Output Feature Dimension {}".format(output_feature.shape))
    glog.info("Start to calculate parameter norms")
    input_fea_mean = np.mean(input_feature, axis=0)
    input_fea_std = np.std(input_feature, axis=0) + STD_EPSILON
    output_fea_mean = np.mean(output_feature, axis=0)
    output_fea_std = np.std(output_feature, axis=0) + STD_EPSILON
    return (
        (input_fea_mean, input_fea_std),
        (output_fea_mean, output_fea_std)
    )
