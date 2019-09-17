#!/usr/bin/env python
""" Control feature visualization related utils """

import glob
import os

import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, FEATURE_NAMES
import fueling.common.logging as logging
import fueling.profiling.feature_extraction.control_feature_extraction_utils as feature_utils


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


def clean_data(data, seq):
    """clean the data until the distribution matches the given standard"""
    length = data.shape[0]
    for i in range(9):
        idx_h = seq[int(length * (1 - 0.05 * i) - 1)]
        idx_l = seq[int(length * (0.05 * i))]
        scope = data[idx_h] - data[idx_l]
        idx_h_partial = seq[int(length * (1 - 0.05 * (i + 1)) - 1)]
        idx_l_partial = seq[int(length * (0.05 * (i + 1)))]
        scope_partial = data[idx_h_partial] - data[idx_l_partial]
        if (scope <= 2 * scope_partial):
            return [0.05 * i, 1 - 0.05 * i]
    return [0.05 * i, 1 - 0.05 * i]


def plot_h5_features_hist(data_rdd):
    """plot the histogram of all the variables in the data array"""
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
    profiling_conf = feature_utils.get_config_control_profiling()
    with PdfPages(pdffile) as pdf:
        for i in range(len(FEATURE_NAMES)):
            if (i < data.shape[1] and
                    (i < FEATURE_IDX["timestamp_sec"] or i > FEATURE_IDX["trajectory_sequence_num"])):
                logging.info('Processing the plots at Column: {}, Feature: {}'
                          .format(i, FEATURE_NAMES[i]))
                if i == FEATURE_IDX["pose_heading_offset"]:
                    data_plot_idx = np.where((data[:, FEATURE_IDX["speed"]] >
                                              profiling_conf.control_metrics.speed_stop) &
                                             (data[:, FEATURE_IDX["curvature_reference"]] <
                                              profiling_conf.control_metrics.curvature_harsh_limit))[0]
                    data_plot = np.take(data[:, i], data_plot_idx, axis=0)
                else:
                    data_plot = data[:, i]
                length = data_plot.shape[0]
                if length > profiling_conf.min_sample_size:
                    seq = np.argsort(data_plot)
                    scope = data_plot[seq[length - 1]] - data_plot[seq[0]]
                    scope_90 = data_plot[seq[int(length * 0.95 - 1)]] - \
                        data_plot[seq[int(length * 0.05)]]
                    logging.info('The data scope is: {} the intermedia-90% data scope is: {}'
                              .format(scope, scope_90))
                    bounds = clean_data(data_plot, seq)
                    if bounds[0] == 0 and bounds[1] == 1:
                        plt.figure(figsize=(4, 3))
                        plt.hist(data_plot, bins=100)
                        plt.xlabel(FEATURE_NAMES[i])
                        plt.ylabel('Sample length')
                        plt.title("Histogram of " + FEATURE_NAMES[i])
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[length - 1]], data_plot[seq[0]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    else:
                        plt.figure(figsize=(4, 3))
                        plt.hist(data_plot[seq[int(length * bounds[0]):int(length * bounds[1] - 1)]],
                                 bins=100)
                        plt.xlabel(FEATURE_NAMES[i])
                        plt.ylabel('Sample length')
                        plt.title("Histogram of " + FEATURE_NAMES[i] + " ("
                                  + str(int(round((bounds[1] - bounds[0]) * 100))) + "% data)")
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[int(length * bounds[1] - 1)]],
                                         data_plot[seq[int(length * bounds[0])]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        plt.figure(figsize=(4, 3))
                        plt.plot(data_plot)
                        plt.ylabel(FEATURE_NAMES[i])
                        plt.xlabel('Sample Number')
                        plt.title("Plot of " + FEATURE_NAMES[i] + " (100% Data)")
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[length - 1]], data_plot[seq[0]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
