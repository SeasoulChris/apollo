#!/usr/bin/env python

import glob
import os
import time

from matplotlib.backends.backend_pdf import PdfPages
import colored_glog as glog
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.common.h5_utils import read_h5
import fueling.common.s3_utils as s3_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils


class MultiCalibrationTableVisualization(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'Multi_Calibration_Table_Visualization')

    def run_test(self):
        origin_dir = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableConf'
        conf_dir = '/apollo/modules/data/fuel/testdata/control/sourceData/OUT'
        self.run(origin_dir, conf_dir)

    def run_prod(self):
        origin_prefix = 'modules/control/data/results/CalibrationTableConf'
        conf_prefix = 'modules/control/data/records/'
        origin_dir = s3_utils.abs_path(origin_prefix)
        conf_dir = s3_utils.abs_path(conf_prefix)
        self.run(origin_dir, conf_dir)

    def run(self, origin_prefix, conf_prefix):
        # PairRDD(vehicle, path_to_vehicle)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            .flatMap(multi_vehicle_utils.get_vehicle)
            .keyBy(lambda vehicle: vehicle)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)), 3)

        # PairRDD((vehicle, 'throttle'), hdf5_file)
        throttle_features = spark_helper.cache_and_log(
            'throttle_hdf5',
            origin_vehicle_dir
            .map(lambda (vehicle, path): (vehicle,
                                          os.path.join(path, 'throttle_calibration_table.pb.txt.hdf5')))
            .mapValues(read_h5))

        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # PairRDD(vehicle, dir_of_vehicle)
            self.to_rdd([conf_prefix])
            .flatMap(multi_vehicle_utils.get_vehicle)
            .keyBy(lambda vehicle: vehicle)
            .mapValues(lambda vehicle: os.path.join(conf_prefix, vehicle))
            .mapValues(multi_vehicle_utils.get_vehicle_param))

        throttle_plots = spark_helper.cache_and_log(
            'throttle_plot',
            vehicle_param_conf
            .mapValues(lambda conf: multi_vehicle_utils.gen_param(conf, 'throttle'))
            .join(throttle_features)
            .map(lambda vehicle_data: multi_vehicle_utils.gen_plot(vehicle_data, origin_prefix, 'throttle')))

        brake_features = spark_helper.cache_and_log(
            'brake_hdf5',
            origin_vehicle_dir
            .map(lambda (vehicle, path): (vehicle,
                                          os.path.join(path, 'brake_calibration_table.pb.txt.hdf5')))
            .mapValues(read_h5))

        brake_plots = spark_helper.cache_and_log(
            'brake_plot',
            vehicle_param_conf
            .mapValues(lambda conf: multi_vehicle_utils.gen_param(conf, 'brake'))
            .join(brake_features)
            .map(lambda vehicle_data: multi_vehicle_utils.gen_plot(vehicle_data, origin_prefix, 'brake')))


if __name__ == '__main__':
    MultiCalibrationTableVisualization().main()
