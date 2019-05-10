#!/usr/bin/env python
""" Control feature visualization related utils """

import glob
import os

import colored_glog as glog
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_NAMES


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

def plot_h5_features_hist(data_rdd):
    """plot the histogram of all the variables in the data array"""
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
        for i in range(len(FEATURE_NAMES)):
            if i < data.shape[1]:
                glog.info('Processing the plots at Column: {}, Feature: {}'
                          .format(i, FEATURE_NAMES[i]))
                length = data.shape[0]
                seq = np.argsort(data[:, i])
                scope = data[seq[length-1], i] - data[seq[0], i]
                scope_90 = data[seq[int(length*0.95)], i] - data[seq[int(length*0.05)], i]
                glog.info('The data scope is: {} the intermedia-90% data scope is: {}'
                          .format(scope, scope_90))
                if scope <= 2*scope_90:
                    plt.figure(figsize=(4, 3))
                    plt.hist(data[:, i], bins='auto')
                    plt.xlabel(FEATURE_NAMES[i])
                    plt.ylabel('Sample length')
                    plt.title("Histogram of " + FEATURE_NAMES[i])
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.text(xmin*0.9+xmax*0.1, ymin*0.1+ymax*0.9,
                             'Maximum = {0:.3f}, Minimum = {1:.3f}'
                             .format(data[seq[length-1], i], data[seq[0], i]),
                             color='red', fontsize=8)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                else:
                    plt.figure(figsize=(4, 3))
                    plt.hist(data[seq[int(length*0.05):int(length*0.95)], i], bins='auto')
                    plt.xlabel(FEATURE_NAMES[i])
                    plt.ylabel('Sample length')
                    plt.title("Histogram of " + FEATURE_NAMES[i] + " (90% Data)")
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.text(xmin*0.9+xmax*0.1, ymin*0.1+ymax*0.9,
                             'Maximum = {0:.3f}, Minimum = {1:.3f}'
                             .format(data[seq[int(length*0.95)], i], data[seq[int(length*0.05)], i]),
                             color='red', fontsize=8)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    plt.figure(figsize=(4, 3))
                    plt.plot(data[:, i])
                    plt.ylabel(FEATURE_NAMES[i])
                    plt.xlabel('Sample Number')
                    plt.title("Plot of " + FEATURE_NAMES[i] + " (100% Data)")
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.text(xmin*0.9+xmax*0.1, ymin*0.1+ymax*0.9,
                             'Maximum = {0:.3f}, Minimum = {1:.3f}'
                             .format(data[seq[length-1], i], data[seq[0], i]),
                             color='red', fontsize=8)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
