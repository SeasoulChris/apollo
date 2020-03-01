#!/usr/bin/env python

import glob
import os
import time

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import segment_index, input_index
from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.logging as logging
import fueling.control.dynamic_model.data_generator.feature_extraction as data_generator
import fueling.control.common.multi_vehicle_plot_utils as multi_vehicle_plot_utils

VEHICLE_ID = feature_config["vehicle_id"]


class DynamicModelDatasetDistribution(BasePipeline):

    def run_test(self):
        # hdf5 data directory
        data_dir = '/fuel/testdata/control/learning_based_model'
        file_dir = os.path.join(
            data_dir, 'hdf5_training/Mkz7/UniformDistributed/forward/*/*/*.hdf5')
        # data_dir = '/fuel/testdata/control/generated/Mkz7/SampleSet'
        # training_dataset = [os.path.join(data_dir, 'hdf5_training/training_test.hdf5')]

        # file path to save visualization results
        output_dir = os.path.join(data_dir, 'evaluation_result')
        timestr = time.strftime('%Y%m%d-%H%M%S')
        file_name = ('dataset_distribution_%s.pdf' % timestr)
        result_file = os.path.join(output_dir, file_name)

        hdf5_file_list = self.to_rdd(glob.glob(file_dir))
        print(hdf5_file_list.collect())
        self.run_internal(hdf5_file_list, result_file)

    def run(self):
        # hdf5 data directory
        # prefix = 'modules/control/data/results/UniformDistributed/Mkz7'
        prefix = 'modules/control/data/results/SampleSet/Mkz7'
        # file path to save visualization results
        output_dir = self.our_storage().abs_path(prefix)
        timestr = time.strftime('%Y%m%d-%H%M%S')
        file_name = ('dataset_distribution_%s.pdf' % timestr)
        result_file = os.path.join(output_dir, file_name)
        logging.info('Result File: %s', result_file)

        hdf5_file_list = self.to_rdd(self.our_storage().list_files(prefix, '.hdf5'))
        self.run_internal(hdf5_file_list, result_file)

    def run_internal(self, hdf5_file_list, result_file):

        data = spark_helper.cache_and_log(
            'plots',
            hdf5_file_list.keyBy(lambda _: 'Mkz7')
            .groupByKey()
            .mapValues(data_generator.generate_segment_from_list))

        plot = spark_helper.cache_and_log(
            'plots',
            data
            .mapValues(lambda features: multi_vehicle_plot_utils
                       .plot_dynamic_model_feature_hist(features, result_file)))


if __name__ == '__main__':
    DynamicModelDatasetDistribution().main()
