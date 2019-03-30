#!/usr/bin/env python
""" Plot the extracted control features in histogram figures """

import glob
import h5py
import numpy as np
import os
import sys
import time

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Need to add the local path into the sys.path if running in the local computer
# sys.path.append('/home/yuwang01/Documents/Apollo_Local/apollo-fuel/')
from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_NAMES


def generate_segments(h5s):
    """generate data segments from all the selected hdf5 files"""
    segments = []
    segments_width = len(FEATURE_NAMES)
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as h5file:
            segment_keys = [n for n in h5file.keys()]
            if len(segment_keys) < 1:
                continue
            for key in range(len(segment_keys)):
                if h5file[segment_keys[key]].shape[0] == segments_width:
                    ds = np.array(h5file[segment_keys[key]])
                    segments.append(ds)
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

def plot_h5_features_hist(data, pdffile):
    """plot the histogram of all the variables in the data array"""
    with PdfPages(pdffile) as pdf:
        for i in range(len(FEATURE_NAMES)):
            plt.figure(figsize=(4, 3))
            plt.hist(data[:, i], bins='auto')
            plt.xlabel(FEATURE_NAMES[i])
            plt.ylabel('Sample Count')
            plt.title("Histogram of the " + FEATURE_NAMES[i])
            plt.tight_layout()
            pdf.savefig()
            plt.close()


if __name__ == "__main__":

    dir_data = './testdata/control/control_profiling/generated/'
    check = os.path.isdir(dir_data)
    hdf5 = glob.glob(dir_data + '*/*.hdf5')

    data_segments = generate_segments(hdf5)
    data_set = generate_data(data_segments)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    pdf_file = dir_data + 'Dataset_Distribution_%s.pdf' % timestr
    plot_h5_features_hist(data_set, pdf_file)
