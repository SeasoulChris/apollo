#!/usr/bin/env python
""" utils for multiple vehicles """
import os
import time

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.control.dynamic_model.conf.model_config import segment_index, input_index
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


DIM_INPUT = 3
cali_input_index = {
    0: 'speed',  # chassis.speed_mps
    1: 'acceleration',
    2: 'control command',
}


def plot_dynamic_model_feature_hist(fearure, result_file):
    logging.info('Total Number of Feature Frames %s' % fearure.shape[0])
    with PdfPages(result_file) as pdf:
        for feature_name in input_index:
            logging.info('feature_name %s' % feature_name)
            # skip if the feature is not in the segment_index list
            if feature_name not in segment_index:
                continue
            feature_index = segment_index[feature_name]
            plt.figure(figsize=(4, 3))
            axes = plt.gca()
            axes.set_ylim([0, 7000])
            # plot the distribution of feature_index column of input data
            plt.hist(fearure[:, feature_index], bins='scott', label='linear')
            plt.title("Histogram of the Feature Input {}".format(feature_name))
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    return result_file


def plot_feature_hist(elem, target_dir):
    vehicle, feature = elem
    # timestr = time.strftime('%Y%m%d-%H%M%S')
    # result_file = os.path.join(target_dir, vehicle, 'Dataset_Distribution_%s.pdf' % timestr)
    result_file = os.path.join(target_dir, vehicle, 'Dataset_Distribution.pdf')
    with PdfPages(result_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4, 3))
            plt.hist(feature[:, j], bins='auto')
            plt.title("Histogram of the " + cali_input_index[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    return result_file


def gen_plot(elem, target_dir, throttle_or_brake):

    (vehicle, (((speed_min, speed_max, speed_segment_num),
                (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha), acc_maxtrix)) = elem

    # timestr = time.strftime('%Y%m%d-%H%M%S')
    # result_file = os.path.join(
    #     target_dir, vehicle, (throttle_or_brake + '_result_%s.pdf' % timestr))
    result_file = os.path.join(
        target_dir, vehicle, (throttle_or_brake + '_result.pdf'))

    cmd_array = np.linspace(cmd_min, cmd_max, num=cmd_segment_num)
    speed_array = np.linspace(speed_min, speed_max, num=speed_segment_num)
    speed_maxtrix, cmd_matrix = np.meshgrid(speed_array, cmd_array)
    grid_array = np.array([[s, c] for s, c in zip(np.ravel(speed_array), np.ravel(cmd_array))])
    with PdfPages(result_file) as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(speed_maxtrix, cmd_matrix, acc_maxtrix,
                        alpha=1, rstride=1, cstride=1, linewidth=0.5, antialiased=True)
        ax.set_xlabel('$speed$')
        ax.set_ylabel('$%s$' % throttle_or_brake)
        ax.set_zlabel('$acceleration$')
        pdf.savefig()
    return result_file
