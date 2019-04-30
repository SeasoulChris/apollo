#!/usr/bin/env python

import glob
import os
import time

import matplotlib
matplotlib.use('Agg')
import pyspark_utils.helper as spark_helper

from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client
import fueling.control.common.multi_vehicle_plot_utils as multi_vehicle_plot_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils


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


class MultiVehicleDataDistribution(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'Multi_Vehicle_Data_Distribution')

    def run_test(self):
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableFeature'

        # PairRDD(vehicle, path_to_vehicle)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            # RDD(input_folder)
            self.to_rdd([origin_prefix])
            # RDD(vehicle)
            .flatMap(os.listdir)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, path_to_vehicle)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        """ origin_prefix/brake_or_throttle/train_or_test/.../*.hdf5 """
        # PairRDD(vehicle, list_of_hdf5_files)
        hdf5_files = spark_helper.cache_and_log(
            'hdf5_files',
            origin_vehicle_dir.mapValues(
                lambda path: glob.glob(os.path.join(path, '*/*/*/*.hdf5'))))

        # origin_prefix: absolute path
        self.run(hdf5_files, origin_prefix)

    def run_prod(self):
        origin_prefix = 'modules/control/data/results/CalibrationTableFeature'

        # PairRDD(vehicle, path_to_vehicle)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            # RDD(abs_input_folder)
            self.to_rdd([bos_client.abs_path(origin_prefix)])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, relative_path_vehicle)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        # PairRDD(vehicle, list_of_hdf5_files)
        hdf5_files = spark_helper.cache_and_log(
            'hdf5_files', origin_vehicle_dir.mapValues(self.list_end_files_prod))

        origin_dir = bos_client.abs_path(origin_prefix)
        self.run(hdf5_files, origin_dir)

    def run(self, hdf5_file, target_dir):
        # PairRDD(vehicle, features)
        features = spark_helper.cache_and_log('features', hdf5_file.mapValues(read_hdf5))
        # PairRDD(vehicle, result_file)
        plots = spark_helper.cache_and_log(
            'plots', features.map(lambda vehicle_feature:
                                  multi_vehicle_plot_utils.plot_feature_hist(vehicle_feature, target_dir)))

    def list_end_files_prod(self, path):
        return self.bos().list_files(path, '.hdf5')


if __name__ == '__main__':
    MultiVehicleDataDistribution().main()
