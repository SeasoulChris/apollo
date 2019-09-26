#!/usr/bin/env python

""" Analyze and plot the vehicle dynamics features. """

import glob
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from scipy.fftpack import fft
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

from fueling.profiling.conf.control_channel_conf import DYNAMICS_FEATURE_IDX, DYNAMICS_FEATURE_NAMES
import fueling.common.logging as logging

# Minimum epsilon value used in compare with zero
MIN_EPSILON = 0.000001


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


def plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1, data_alivezone,
                     feature, title_addon, text_addon):
    """ control actions x,y - time """
    plt.plot(data_plot_x0, data_plot_y0, label="command", linewidth=0.5)
    plt.plot(data_plot_x1, data_plot_y1, label="action", linewidth=0.5)
    plt.xlabel('timestamp_sec (relative to t0) /sec')
    plt.ylabel(feature + ' commands and measured outputs /%')
    plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " and " +
              DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " (" + title_addon + ")",
              fontsize=10)
    plt.legend(fontsize=6)
    xmin, xmax, ymin, ymax = plt.axis()
    data_est_y0 = np.take(data_plot_y0, data_alivezone, axis=0)
    data_est_y1 = np.take(data_plot_y1, data_alivezone, axis=0)
    plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
             'Max Devidation = {0:.3f}, \nStandard Deviation = {1:.3f}'
             .format(np.amax(np.abs(data_est_y1 - data_est_y0)),
                     np.std(data_est_y1 - data_est_y0)),
             color='red', fontsize=8)
    plt.text(xmin * 0.5 + xmax * 0.5, ymin * 0.5 + ymax * 0.5, text_addon,
             color='red', fontsize=6)
    plt.tight_layout()


def plot_act_vs_cmd(data_plot_x, data_plot_y, data_alivezone, feature, title_addon, polyfit):
    """ control actions x - y """
    data_est_x = np.take(data_plot_x, data_alivezone, axis=0)
    data_est_y = np.take(data_plot_y, data_alivezone, axis=0)
    plt.plot(data_plot_x, data_plot_y, '.', markersize=2)
    plt.axis('equal')
    plt.xlabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]])
    plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]])
    plt.title(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]] + " vs " +
              DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " (" + title_addon + ")",
              fontsize=10)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
             'Max Devidation = {0:.3f}, \nStandard Deviation = {1:.3f}'
             .format(np.amax(np.abs(data_est_y - data_est_x)),
                     np.std(data_est_y - data_est_x)),
             color='red', fontsize=6)
    plt.tight_layout()
    var_polyfit = [1.0, 0.0]
    if polyfit:
        var_polyfit = np.polyfit(data_est_x, data_est_y, 1)
        line_polyfit = np.poly1d(var_polyfit)
        fit_plot_x = np.array([np.amin(data_plot_x), np.amax(data_plot_x)])
        fit_plot_y = line_polyfit(fit_plot_x)
        plt.plot(fit_plot_x, fit_plot_y, '--r')
        plt.text(xmin * 0.4 + xmax * 0.6, ymin * 0.8 + ymax * 0.2,
                 'Polyfit: slope = {0:.3f}, \nPolyfit: bias = {1:.3f}'
                 .format(var_polyfit[0], var_polyfit[1]),
                 color='red', fontsize=6)
    return var_polyfit


def plot_xcorr(data_plot_x, data_plot_y, data_alivezone, feature):
    """ cross-correlation x - y """
    data_xcorr_x = np.take(data_plot_x, data_alivezone, axis=0)
    data_xcorr_y = np.take(data_plot_y, data_alivezone, axis=0)
    lags = plt.xcorr(data_xcorr_x, data_xcorr_y, usevlines=True, maxlags=50,
                     normed=True, lw=0.5, linestyle='solid', color='blue')
    plt.grid(True)
    plt.xlabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " delay frames")
    plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " cross-correlation")
    plt.title("Cross-correlation " + DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
              + " vs " + DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]], fontsize=10)
    lag_frame = lags[0][np.argmax(lags[1])]
    plt.plot(lag_frame, 1.0, "o", markersize=2, markerfacecolor='r', markeredgecolor='b')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(lag_frame + 3, ymin * 0.05 + ymax * 0.95,
             'Delay Frame = {}'.format(lag_frame),
             color='red', fontsize=6)
    plt.tight_layout()
    print('The estimated delay frame number is: ', lag_frame)
    return lag_frame


def plot_fft(data_plot_x, data_plot_y, data_alivezone, feature, title_addon):
    """ fast fourier transformation """
    # Number of sample points
    L = data_plot_x.shape[0]
    N = np.power(2, np.ceil(np.log2(abs(L)))).astype(int)
    # Sample spacing
    T = np.median(np.diff(data_plot_x, axis=0), axis=0)
    # Fourier Transformation
    data_fft_y = fft(mlab.detrend_mean(data_plot_y[data_alivezone], axis=0), n=N)
    data_fft_x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(data_fft_x, 2.0 / N * np.abs(data_fft_y[0: N // 2]))
    plt.xlim([-1, 4])
    plt.grid(True)
    plt.xlabel('Frequency /Hz')
    plt.ylabel(DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] + " Power Spectrum")
    plt.title("Frequency Spectrum of " + DYNAMICS_FEATURE_NAMES[DYNAMICS_FEATURE_IDX[feature]] +
              " (" + title_addon + ")", fontsize=10)
    plt.tight_layout()


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
        pdffile = os.path.join(dir_data, vehicle_controller +
                               'control_data_visualization_time_domain.pdf')
    else:
        pdffile = os.path.join(dir_data, 'control_data_visualization_time_domain.pdf')
    with PdfPages(pdffile) as pdf:
        data = data[np.argsort(data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]])]
        plot_features = ["throttle", "brake", "steering", "acceleration"]
        for feature in plot_features:
            if feature is "throttle":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            elif feature is "brake":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            elif feature is "steering":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            elif feature is "acceleration":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            # Raw data plots and analysis
            title_addon = "raw data"
            data_plot_x0 = (data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[:, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            if feature is "steering" or feature is "acceleration":
                data_plot_y0 = (data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]- bias_y) / slope_y
            else:
                data_plot_y0 = np.maximum(0, (data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                                 - bias_y) / slope_y)
            data_plot_y1 = data[:, DYNAMICS_FEATURE_IDX[feature]]
            data_alivezone_y0 = np.where(np.abs(data_plot_y0) > MIN_EPSILON)[0]
            text_addon = "cmd-act scaling slope = {0:.3f}, \ncmd-act scaling bias = {1:.3f}, \
                          \ncmd-act shift frame = {2:.3f}".format(slope_y, bias_y, delay_frame)
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1,
                             data_alivezone_y0, feature, title_addon, text_addon)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, data_alivezone_y0,
                                          feature, title_addon, False)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            delay_frame = plot_xcorr(data_plot_y0, data_plot_y1, data_alivezone_y0, feature)
            pdf.savefig()
            plt.close()
            # Shifted data by estimating delay frame number
            title_addon = "aligned data"
            data_plot_x0 = (data[0:-1 + delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0 - delay_frame, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            if feature is "steering" or feature is "acceleration":
                data_plot_y0 = (data[0:-1 + delay_frame, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                - bias_y) / slope_y
            else:
                data_plot_y0 = np.maximum(0.0, (data[0:-1 + delay_frame,
                                                     DYNAMICS_FEATURE_IDX[feature + "_cmd"]]
                                                - bias_y) / slope_y)
            data_plot_y1 = data[0 - delay_frame:-1, DYNAMICS_FEATURE_IDX[feature]]
            data_alivezone_y0 = np.where(np.abs(data_plot_y0) > MIN_EPSILON)[0]
            text_addon = "cmd-act scaling slope = {0:.3f}, \ncmd-act scaling bias = {1:.3f}, \
                          \ncmd-act shift frame = {2:.3f}".format(slope_y, bias_y, delay_frame)
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1,
                             data_alivezone_y0, feature, title_addon, text_addon)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, data_alivezone_y0, feature,
                                          title_addon, True)
            pdf.savefig()
            plt.close()
            # Scaled data by fitting the data curve
            title_addon = "scaled data"
            if feature is "steering" or feature is "acceleration":
                data_plot_y0 = data_plot_y0 * var_polyfit[0] + var_polyfit[1]
            else:
                data_plot_y0[data_alivezone_y0] = np.maximum(0.0, data_plot_y0[data_alivezone_y0]
                                                                  * var_polyfit[0] + var_polyfit[1])
            bias_y += slope_y * var_polyfit[1]
            slope_y *= var_polyfit[0]
            text_addon = "cmd-act scaling slope = {0:.3f}, \ncmd-act scaling bias = {1:.3f}, \
                          \ncmd-act shift frame = {2:.3f}".format(slope_y, bias_y, delay_frame)
            plt.figure(figsize=(4, 4))
            plot_ctl_vs_time(data_plot_x0, data_plot_x1, data_plot_y0, data_plot_y1,
                             data_alivezone_y0, feature, title_addon, text_addon)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            var_polyfit = plot_act_vs_cmd(data_plot_y0, data_plot_y1, data_alivezone_y0,
                                          feature, title_addon, True)
            pdf.savefig()
            plt.close()


def plot_h5_features_freq(data_rdd):
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
        pdffile = os.path.join(dir_data, vehicle_controller +
                                         'control_data_visualization_frequency_domain.pdf')
    else:
        pdffile = os.path.join(dir_data, 'control_data_visualization_frequency_domain.pdf')
    with PdfPages(pdffile) as pdf:
        data = data[np.argsort(data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]])]
        plot_features = ["acceleration", "steering", ]
        for feature in plot_features:
            if feature is "throttle":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            elif feature is "acceleration":
                slope_y = 1.0
                bias_y = 0.0
                delay_frame = 0
            # Raw data plots and analysis
            title_addon = "raw data"
            data_plot_x0 = (data[:, DYNAMICS_FEATURE_IDX["timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_x1 = (data[:, DYNAMICS_FEATURE_IDX["chassis_timestamp_sec"]] -
                            data[0, DYNAMICS_FEATURE_IDX["timestamp_sec"]])
            data_plot_y0 = (data[:, DYNAMICS_FEATURE_IDX[feature + "_cmd"]]- bias_y) / slope_y
            data_plot_y1 = data[:, DYNAMICS_FEATURE_IDX[feature]]
            data_alivezone_y0 = np.where(np.abs(data_plot_y0) > MIN_EPSILON)[0]
            plt.figure(figsize=(4, 4))
            plot_fft(data_plot_x0, data_plot_y0, data_alivezone_y0, feature + "_cmd", title_addon)
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 4))
            plot_fft(data_plot_x1, data_plot_y1, data_alivezone_y0, feature, title_addon)
            pdf.savefig()
            plt.close()
