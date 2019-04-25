#!/usr/bin/env python

import glob
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import colored_glog as glog
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index
import fueling.common.s3_utils as s3_utils
import fueling.control.dynamic_model.data_generator.data_generator as data_generator

VEHICLE_ID = 'Mkz7'


def plot_feature_hist(fearure, result_file):
    with PdfPages(result_file) as pdf:
        for feature_name in input_index:
                # skip if the feature is not in the segment_index list
                if feature_name not in segment_index:
                    continue
                feature_index = segment_index[feature_name]
                plt.figure(figsize=(4,3))
                # plot the distribution of feature_index column of input data
                plt.hist(fearure[:, feature_index], bins ='auto', label='linear')
                plt.title("Histogram of the Feature Input {}".format(feature_name))
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
    return result_file


class DynamicModelDatasetDistribution(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'Dynamic_Model_Data_Distribution')

    def run_test(self):
        # hdf5 data directory
        data_dir = '/apollo/modules/data/fuel/testdata/control/learning_based_model'
        training_dataset = [os.path.join(data_dir, 'hdf5_training/training_test.hdf5')]

        # file path to save visualization results
        output_dir = os.path.join(data_dir, 'evaluation_result')
        file_name = 'dataset_distribution_%s.pdf' % VEHICLE_ID
        result_file = os.path.join(output_dir, file_name)

        # RDD(file_path) for training dataset.
        hdf5_rdd = self.to_rdd(training_dataset)
        if hdf5_rdd.count() != 0:
            hdf5_file_list = hdf5_rdd.collect()
            self.run(hdf5_file_list, result_file)
        else:
            glog.error('No hdf5 files are found')


    def run_prod(self):
        # hdf5 data directory
        bucket = 'apollo-platform'
        prefix = 'modules/control/learning_based_model/hdf5_training/Mkz7/UniformDistributed'

        # file path to save visualization results
        output_dir = s3_utils.abs_path(
            'modules/control/learning_based_model/evaluation_result/')
        file_name = 'dataset_distribution_%s.pdf' % VEHICLE_ID
        result_file = os.path.join(output_dir, file_name)

        # RDD(file_path) for training dataset
        hdf5_rdd = s3_utils.list_files(bucket, prefix, '.hdf5').cache()

        if hdf5_rdd.count() != 0:
            hdf5_file_list = hdf5_rdd.collect()
            self.run(hdf5_file_list, result_file)
        else:
            glog.error('No hdf5 files are found')

    def run(self, hdf5_file_list, result_file):
        features = data_generator.generate_segment_from_list(hdf5_file_list)
        plot_feature_hist(features, result_file)


if __name__ == '__main__':
    DynamicModelDatasetDistribution().main()
