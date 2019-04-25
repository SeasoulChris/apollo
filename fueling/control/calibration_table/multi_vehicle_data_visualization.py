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
import fueling.common.s3_utils as s3_utils
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


def list_end_files_prod(path):
    bucket = 'apollo-platform'
    return s3_utils.list_files(bucket, path, '.hdf5').collect()


class MultiVehicleDataDistribution(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'Multi_Vehicle_Data_Distribution')

    def run_test(self):
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableFeature'

        # PairRDD(vehicle, path_to_vehicle)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir', multi_vehicle_utils.get_vehicle_rdd(origin_prefix), 3)

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
            self.to_rdd([os.path.join(s3_utils.BOS_MOUNT_PATH, origin_prefix)])
            .flatMap(os.listdir)
            .keyBy(lambda vehicle: vehicle)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        # PairRDD(vehicle, list_of_hdf5_files)
        hdf5_files = spark_helper.cache_and_log(
            'hdf5_files',
            origin_vehicle_dir.mapValues(list_end_files_prod))

        origin_dir = s3_utils.abs_path(origin_prefix)
        self.run(hdf5_files, origin_dir)

    def run(self, hdf5_file, target_dir):
        # PairRDD(vehicle, features)
        features = spark_helper.cache_and_log('features', hdf5_file.mapValues(read_hdf5))
        # PairRDD(vehicle, result_file)
        plots = spark_helper.cache_and_log(
            'plots', features.map(lambda vehicle_feature:
                                  multi_vehicle_utils.plot_feature_hist(vehicle_feature, target_dir)))


if __name__ == '__main__':
    MultiVehicleDataDistribution().main()
