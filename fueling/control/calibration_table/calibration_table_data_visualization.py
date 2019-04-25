#!/usr/bin/env python

import glob
import os
import time

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import colored_glog as glog
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fueling.common.base_pipeline import BasePipeline


def read_hdf5(hdf5_file_list):
    """
    load h5 file to a numpy array
    """
    segment = None
    for filename in hdf5_file_list:
        with h5py.File(filename, 'r') as fin:
            for value in fin.values():
                if segment is None:
                    segment = np.array(value)
                else:
                    segment = np.concatenate((segment, np.array(value)), axis=0)
    return segment


DIM_INPUT = 3
input_index = {
    'speed': 0,  # chassis.speed_mps
    'acceleration': 1,
    'control command': 2,
}


def plot_feature_hist(fearure, result_file):
    with PdfPages(result_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4, 3))
            plt.hist(fearure[:, j], bins='auto')
            plt.title("Histogram of the " + list(input_index)[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    return result_file


class CalibrationTableDataDistribution(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'Calibration_Table_Data_Distribution')

    def run_test(self):
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated/for_plot'
        timestr = time.strftime('%Y%m%d-%H%M%S')
        result_file = os.path.join(origin_prefix, 'Dataset_Distribution_%s.pdf' % timestr)
        hdf5_file_list = glob.iglob(os.path.join(origin_prefix, '*.hdf5'))

        self.run(hdf5_file_list, result_file)

    def run_prod(self):
        origin_prefix = 'modules/control/CalibrationTable'

        timestr = time.strftime('%Y%m%d-%H%M%S')
        target_prefix = '/mnt/bos/modules/control/CalibrationTable'
        result_file = os.path.join(target_prefix, 'Dataset_Distribution_%s.pdf' % timestr)

        hdf5_file_list = self.bos().list_files(origin_prefix, '.hdf5')
        if hdf5_file_list:
            self.run(hdf5_file_list, result_file)
        else:
            glog.error('No hdf5 files are found')

    def run(self, hdf5_file_list, result_file):
        features = read_hdf5(hdf5_file_list)
        plot_feature_hist(features, result_file)


if __name__ == '__main__':
    CalibrationTableDataDistribution().main()
