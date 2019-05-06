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
    data = segments[0]
    for i in range(1, len(segments)):
        data = np.vstack([data, segments[i]])
    print('Data_Set length is: ', len(data))
    return data

def plot_h5_features_hist(data_rdd):
    """plot the histogram of all the variables in the data array"""
    # PairRDD(target_dir, data_array)
    dir_data, data = data_rdd
    pdffile = dir_data + '/control_data_visualization.pdf'
    with PdfPages(pdffile) as pdf:
        for i in range(len(FEATURE_NAMES)):
            if i < data.shape[1]:
                plt.figure(figsize=(4, 3))
                plt.hist(data[:, i], bins='auto')
                plt.xlabel(FEATURE_NAMES[i])
                plt.ylabel('Sample Count')
                plt.title("Histogram of the " + FEATURE_NAMES[i])
                plt.tight_layout()
                pdf.savefig()
                plt.close()
