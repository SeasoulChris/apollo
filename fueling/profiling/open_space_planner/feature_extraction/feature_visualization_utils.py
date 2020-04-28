#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os

import fueling.common.logging as logging

from fueling.profiling.common.stats_utils import IQR_outlier_filter
from fueling.profiling.conf.open_space_planner_conf import TRAJECTORY_FEATURE_NAMES

def plot(data_rdd):
    output_dir, data = data_rdd
    hist_pdf_file = os.path.join(output_dir, 'feature_visualization.pdf')
    time_pdf_file = os.path.join(output_dir, 'feature_timeline_visualization.pdf')
    with PdfPages(hist_pdf_file) as hist_pdf, PdfPages(time_pdf_file) as time_pdf:
        timestamps = data[:, 0]
        for (i, feature_name) in enumerate(TRAJECTORY_FEATURE_NAMES):
            raw_samples = data[:, i]
            raw_length = raw_samples.shape[0]
            clean_idx, outlier_idx = IQR_outlier_filter(raw_samples)
            samples = raw_samples[clean_idx]
            length = samples.shape[0]
            logging.info(F'Generating plot for Feature {i}: {feature_name}, '
                         F'{length}/{raw_length} samples used for histogram.')
            
            # Histogram
            plt.figure(figsize=(4, 3))
            plt.hist(samples, bins=100)
            plt.xlabel(feature_name)
            plt.ylabel('Sample length')
            plt.title(F'Histogram of {feature_name} '
                      F'({str(int(round(length / raw_length * 100)))}% Data)')
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     F'Max = {np.amax(samples):.3f}, Min = {np.amin(samples):.3f}',
                     color='red', fontsize=8)
            plt.tight_layout()
            hist_pdf.savefig()
            plt.close()

            # Plot against sample number
            plt.figure(figsize=(4, 3))
            plt.plot(raw_samples)
            plt.ylabel(feature_name)
            plt.xlabel('Sample Number')
            plt.title(F'Plot of {feature_name} (100% Data)')
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     F'Max = {np.amax(raw_samples):.3f}, Min = {np.amin(raw_samples):.3f}',
                     color='red', fontsize=8)
            plt.tight_layout()
            hist_pdf.savefig()
            plt.close()

            # Plot against timeline
            plt.figure(figsize=(10, 3))
            plt.plot(timestamps[clean_idx], samples, '+', markersize=2, alpha=0.3)
            if len(outlier_idx) > 0:
                plt.plot(timestamps[outlier_idx], raw_samples[outlier_idx], 'r+', markersize=2,
                         alpha=0.3)
            plt.ylabel(feature_name)
            plt.xlabel('Time')
            plt.title(F'Plot of {feature_name} (100% Data)')
            plt.tight_layout()
            time_pdf.savefig()
            plt.close()
