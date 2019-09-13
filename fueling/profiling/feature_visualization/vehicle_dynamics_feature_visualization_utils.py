#!/usr/bin/env python

""" Analyze and plot the vehicle dynamics features. """

import glob
import os

import colored_glog as glog
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from fueling.profiling.conf.control_channel_conf import DYNAMICS_FEATURE_IDX, DYNAMICS_FEATURE_NAMES

def generate_segments(h5s):
    """generate data segments from all the selected hdf5 files"""
    segments = []
    if not h5s:
        glog.warn('No hdf5 files found under the targeted path.')
        return segments
    for h5 in h5s:
        glog.info('Loading {}'.format(h5))
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
        glog.warn('No segments from hdf5 files found under the targetd path.')
        return data
    data.append(segments[0])
    for i in range(1, len(segments)):
        data = np.vstack([data, segments[i]])
    print('Data_Set length is: ', len(data))
    return data


def plot_h5_features_time(data_rdd):
    """plot the time-domain data of all the variables in the data array"""
    # PairRDD(target_dir, data_array)
    dir_data, data = data_rdd
    if len(data) == 0:
        glog.warn('No data from hdf5 files can be visualized under the targetd path {}'
                  .format(dir_data))
        return
    grading_dir = glob.glob(os.path.join(dir_data, '*grading.txt'))
    if grading_dir:
        vehicle_controller = os.path.basename(grading_dir[0]).replace(
            'control_performance_grading.txt', '')
        pdffile = os.path.join(dir_data, vehicle_controller + 'control_data_visualization.pdf')
    else:
        pdffile = os.path.join(dir_data, 'control_data_visualization.pdf')
    with PdfPages(pdffile) as pdf:
        data = data[np.argsort(data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]])]
        plot_features = ["throttle", "brake", "steering"]
        for feature in plot_features:
            data_plot_x0 = (data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[:, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_y0 = data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
            data_plot_y1 = data[:, DYNAMICS_FEATURE_IDX[feature]]
            # control actions x,y-time
            plt.figure(figsize=(4, 4))
            plt.plot(data_plot_x0, data_plot_y0, data_plot_x1, data_plot_y1)
            plt.xlabel('timestamp_sec (relative to t0)')
            plt.ylabel(feature + ' commands and measured outputs')
            plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " and " +
                      DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     'Maximum = {0:.3f}, Minimum = {1:.3f}'
                     .format(np.amax(data_plot_y1), np.amin(data_plot_y1)),
                     color='red', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            # control actions x-y
            data_plot_x = data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
            data_plot_y = data[:, DYNAMICS_FEATURE_IDX[feature]]
            plt.figure(figsize=(4, 4))
            plt.plot(data_plot_x, data_plot_y, '.')
            plt.axis('equal')
            plt.xlabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]])
            plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
            plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " vs " +
                      DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     'Maximum = {0:.3f}, Minimum = {1:.3f}'
                     .format(np.amax(data_plot_y1), np.amin(data_plot_y1)),
                     color='red', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
