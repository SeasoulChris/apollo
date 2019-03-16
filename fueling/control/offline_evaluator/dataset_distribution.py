#!/usr/bin/env python

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import os
import glob
import math
import sys
import time

import h5py
import numpy as np
import google.protobuf.text_format as text_format
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
from scipy.signal import savgol_filter

from features.parameters_training import dim

# Constants
dim_input = dim["pose"] + dim["incremental"] + dim["control"] # accounts for mps
dim_output = dim["incremental"] # the speed mps is also output
input_features = ["speed mps","speed incremental","angular incremental","throttle","brake","steering"]

# TODO use argparse for hdf5 file dir
def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as f:
            names = [n for n in f.keys()]
            if len(names) < 1:
                continue
            for i in range(len(names)):
                ds = np.array(f[names[i]])
                segments.append(ds)
    print('Segments count: ', len(segments))
    return segments

def generate_data(segments):
    total_len = 0
    for i in range(len(segments)):
        total_len += (segments[i].shape[0] - 2)
    print "total_len = ", total_len
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    print "Y size = ", Y.shape
    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(1, segment.shape[0] - 1):
            X[i, 0] = segment[k - 1, 14] #speed mps
            X[i, 1] = (segment[k - 1, 8] * np.cos(segment[k-1,0]) +
                segment[k-1,9] * np.sin(segment[k-1,0]))  #acc
            X[i, 2] = segment[k - 1, 13] #angular speed
            X[i, 3] = segment[k - 1, 15] #control from chassis
            X[i, 4] = segment[k - 1, 16] #control from chassis
            X[i, 5] = segment[k - 1, 17] #control from chassis
            Y[i, 0] = (segment[k, 8] * np.cos(segment[k,0]) +
                segment[k,9] * np.sin(segment[k,0])) #acc next
            Y[i, 1] = segment[k, 13] #angular speed next
            i += 1
    X[:,1] = savgol_filter(X[:,1], 51, 3) # window size 51, polynomial order 3
    Y[:,0] = savgol_filter(Y[:,0], 51, 3) # window size 51, polynomial order 3
    return X, Y

def plot_H5_features_hist(X):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pdf_file = '/mnt/bos/modules/control/evaluation_result/Dataset_Distribution_%s.pdf' % timestr
    with PdfPages(pdf_file) as pdf:
        for j in range(dim_input):
            plt.figure(figsize=(4,3))
            plt.hist(X[:,j],bins ='auto')
            plt.title ("Histogram of the "+input_features[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

if __name__ == "__main__":

    # NOTE: YOU MAY NEED TO CHANGE THIS PATH ACCORDING TO YOUR ENVIRONMENT
    hdf5 = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_training/*.hdf5')
    print "hdf5 files are :"
    print hdf5

    segments = generate_segments(hdf5)
    X, Y = generate_data(segments)

    plot_H5_features_hist(X)

    print "X shape = ", X.shape
    print "Y shape = ", Y.shape
