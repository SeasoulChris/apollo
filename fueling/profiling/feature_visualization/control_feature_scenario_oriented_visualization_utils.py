#!/usr/bin/env python

""" Analyze and plot the scenario-oriented control features. """

import glob
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, FEATURE_NAMES
import fueling.common.logging as logging

# Minimum epsilon value used in compare with zero
MIN_EPSILON = 0.000001
# Minimum timestamp used for feature plotting
MIN_TIME = 0.0
# Maximum timestamp used for feature plotting
MAX_TIME = 60.0


def generate_segments(h5s):
    """generate data segments from all the selected hdf5 files"""
    segments = []
    if not h5s:
        logging.warning('No hdf5 files found under the targeted path.')
        return segments
    for h5 in h5s:
        logging.info('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as h5file:
            for value in h5file.values():
                segments.append(np.array(value))
    print('Segments width is: ', len(segments[0]))
    print('Segments length is: ', len(segments))
    return segments


def generate_data(segments):
    """generate data array from the given data segments"""
    data = []
    if not segments:
        logging.warning('No segments from hdf5 files found under the targetd path.')
        return data
    data.append(segments[0])
    for i in range(1, len(segments)):
        data = np.vstack([data, segments[i]])
    print('Data_Set length is: ', len(data))
    return data


def plot_feature_vs_time(data_plot_x, data_plot_y, label_plot, feature, title_addon):
    """ control feature y v.s. time x """
    for i in range(0, len(data_plot_x)):
        plt.plot(data_plot_x[i], data_plot_y[i], label=label_plot[i], linewidth=0.2)
    plt.xlabel('timestamp_sec /sec')
    plt.ylabel(feature)
    plt.title(FEATURE_NAMES[FEATURE_IDX[feature]] + " (" + title_addon + ")", fontsize=10)
    plt.legend(fontsize=6)
    plt.tight_layout()


def plot_h5_features_per_scenario(data_list):
    """plot the scenario-oriented data of all the selected variables in the data array"""
    # Initialize the data structure
    dir_data = [];
    data = [];
    for list in data_list:
        dir, array = list
        if len(array) == 0:
            logging.warning('No data from hdf5 files can be visualized under the path {}'
                            .format(dir))
            return
        lower_condition = (array[:, FEATURE_IDX["timestamp_sec"]] >= MIN_TIME)
        upper_condition = (array[:, FEATURE_IDX["timestamp_sec"]] <= MAX_TIME)
        array = np.take(array, np.where(lower_condition & upper_condition)[0], axis=0)
        data.append(array[np.argsort(array[:, FEATURE_IDX["timestamp_sec"]])])
        dir_data.append(dir)
    # Initialize the output file path and name
    grading_dir = glob.glob(os.path.join(dir_data[0], '*grading.txt'))
    if grading_dir:
        vehicle_controller = os.path.basename(grading_dir[0]).replace(
            'control_performance_grading.txt', '')
        pdffile = os.path.join(dir_data[0], vehicle_controller +
                               'control_data_visualization_per_scenario.pdf')
    else:
        pdffile = os.path.join(dir_data[0], 'control_data_visualization_per_scenario.pdf')
    # Plot the selected features
    with PdfPages(pdffile) as pdf:
        plot_features = ["station_error", "speed_error", "lateral_error", "heading_error",
                         "steering_chassis"]
        for feature in plot_features:
            title_addon = str(len(data)) + " test cases"
            data_plot_x = [array[:, FEATURE_IDX["timestamp_sec"]] for array in data]
            data_plot_y = [array[:, FEATURE_IDX[feature]] for array in data]
            label_plot = [os.path.basename(dir) for dir in dir_data]
            plt.figure(figsize=(4, 4))
            plot_feature_vs_time(data_plot_x, data_plot_y, label_plot, feature, title_addon)
            pdf.savefig()
            plt.close()
