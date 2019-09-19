#!/usr/bin/env python

""" Analyze and plot the vehicle dynamics features. """

import glob
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fueling.profiling.conf.control_channel_conf import DYNAMICS_FEATURE_IDX, DYNAMICS_FEATURE_NAMES
import fueling.common.logging as logging


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

def plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1, feature, status):
    """ control actions x,y - time """
    plt.plot(data_plot_x0, data_plot_y0, data_plot_x1, data_plot_y1, linewidth=0.5)
    plt.xlabel('timestamp_sec (relative to t0)')
    plt.ylabel(feature + ' commands and measured outputs')
    plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " and " +
              DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " (" + status + ")")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
             'Maximum = {0:.3f}, Minimum = {1:.3f}'
             .format(np.amax(data_plot_y1), np.amin(data_plot_y1)),
             color='red', fontsize=8)
    plt.tight_layout()

def plot_act_vs_cmd(data_plot_x, data_plot_y, feature, status, polyfit):
    """ control actions x - y """
    plt.plot(data_plot_x, data_plot_y, '.', markersize=2)
    plt.axis('equal')
    # plt.xlim(0, 40)
    # plt.ylim(0, 40)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]])
    plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
    plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " vs " +
              DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " (" + status + ")")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
             'Maximum = {0:.3f}, Minimum = {1:.3f}'
             .format(np.amax(data_plot_y), np.amin(data_plot_y)),
             color='red', fontsize=6)
    plt.tight_layout()
    var_polyfit = [1.0, 0.0]
    if polyfit:
        var_polyfit = np.polyfit(data_plot_x, data_plot_y, 1)
        line_polyfit = np.poly1d(var_polyfit)
        fit_plot_x = np.array([np.amin(data_plot_x), np.amax(data_plot_x)])
        fit_plot_y = line_polyfit(fit_plot_x)
        plt.plot(fit_plot_x, fit_plot_y, '--r')
        plt.text(xmin * 0.6 + xmax * 0.4, ymin * 0.8 + ymax * 0.2,
                 'Polyfit: slope = {0:.3f}; bias = {1:.3f}'.format(var_polyfit[0], var_polyfit[1]),
                 color='red', fontsize=6)
    return var_polyfit

def plot_xcorr(data_plot_x, data_plot_y, feature):
    """ cross-correlation x - y """
    lags = plt.xcorr(data_plot_x, data_plot_y, usevlines=True, maxlags=50,
                     normed=True, lw=0.5, linestyle='solid', color='blue')
    plt.grid(True)
    plt.xlabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " delay frames")
    plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " cross-correlation")
    plt.title("Cross-correlation " + DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
              + " vs " + DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
    lag_frame = lags[0][np.argmax(lags[1])]
    plt.plot(lag_frame, 1.0, "o", markersize=2, markerfacecolor='r', markeredgecolor='b')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(lag_frame + 3, ymin * 0.05 + ymax * 0.95,
             'Delay Frame = {}'.format(lag_frame),
             color='red', fontsize=6)
    plt.tight_layout()
    print('"The estimated delay frame number is: ', lag_frame)
    return lag_frame

def plot_h5_features_time(data_rdd):
    """plot the time-domain data of all the variables in the data array"""
    # PairRDD(target_dir, data_array)
    dir_data, data = data_rdd
    if len(data) == 0:
        logging.warning('No data from hdf5 files can be visualized under the targetd path {}'
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
            if feature is "throttle":
                slope_y0 = 1.0
                bias_y0 = -8.0
            elif feature is "brake":
                slope_y0 = 1.0
                bias_y0 = 0.0
            elif feature is "steering":
                slope_y0 = 1.0
                bias_y0 = 0.0
            # Raw data plots and analysis
            status = "raw data"
            data_plot_x0 = (data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[:, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_y0 = np.maximum(0, data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                         * slope_y0 + bias_y0)
            data_plot_y1 = data[:, DYNAMICS_FEATURE_IDX[feature]]
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1, feature, status)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, feature, status, False)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            delay_frame = plot_xcorr(data_plot_y0, data_plot_y1, feature)
            pdf.savefig()
            plt.close()
            # Shifted data by estimating delay frame number
            status = "aligned data"
            data_plot_x0 = (data[0:-1 + delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0 - delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_y0 = np.maximum(0, data[0:-1 + delay_frame,
                                              DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                              * slope_y0 + bias_y0)
            data_plot_y1 = data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX[feature]]
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1, feature, status)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, feature, status, True)
            pdf.savefig()
            plt.close()
            # Scaled data by fitting the data curve
            status = "scaled data"
            slope_y0 *= var_polyfit[0]
            bias_y0 = bias_y0 * var_polyfit[0] + var_polyfit[1]
            data_plot_x0 = (data[0:-1 + delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0 - delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_y0 = np.maximum(0, data[0:-1 + delay_frame,
                                              DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                              * slope_y0 + bias_y0)
            data_plot_y1 = data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX[feature]]
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1, feature, status)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, feature, status, True)
            pdf.savefig()
            plt.close()
