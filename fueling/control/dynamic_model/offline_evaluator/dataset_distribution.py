#!/usr/bin/env python

import os
import glob
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fueling.control.dynamic_model.conf.model_config import input_index, feature_config

# Constants
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]

def plot_feature_hist(fearure):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pdf_file = '/mnt/bos/modules/control/evaluation_result/Dataset_Distribution_%s.pdf' % timestr
    with PdfPages(pdf_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4,3))
            plt.hist(fearure[:,j],bins ='auto')
            plt.title ("Histogram of the " + list(input_index)[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
