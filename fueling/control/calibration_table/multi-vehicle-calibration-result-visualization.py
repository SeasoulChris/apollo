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
import fueling.common.bos_client as bos_client
import fueling.control.common.multi_vehicle_plot_utils as multi_vehicle_plot_utils
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
        origin_dir = bos_client.abs_path(origin_prefix)
        conf_dir = bos_client.abs_path(conf_prefix)
        self.run(origin_dir, conf_dir)

    def run(self, origin_prefix, conf_prefix):
        # PairRDD(vehicle, path_to_vehicle)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            # RDD(abs_path_to_folder)
            self.to_rdd([origin_prefix])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, abs_path_to_vehicle_folder)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)), 3)

        # PairRDD((vehicle, 'throttle'), hdf5_file)
        throttle_features = spark_helper.cache_and_log(
            'throttle_hdf5',
            origin_vehicle_dir
            # PairRDD(vehicle, abs_path_to_hdf5)
            .mapValues(lambda path: os.path.join(path, 'throttle_calibration_table.pb.txt.hdf5'))
            # PairRDD(vehicle, data)
            .mapValues(read_h5))

        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # RDD(abs_path_to_folder)
            self.to_rdd([conf_prefix])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, abs_path_to_vehicle_folder)
            .mapValues(lambda vehicle: os.path.join(conf_prefix, vehicle))
            # PairRDD(vehicle, VEHICLE_PARAM_CONF.vehicle_param)
            .mapValues(multi_vehicle_utils.get_vehicle_param))

        throttle_plots = spark_helper.cache_and_log(
            'throttle_plot',
            # PairRDD(vehicle, VEHICLE_PARAM_CONF.vehicle_param)
            vehicle_param_conf
            # PairRDD(vehicle, throttle_param)
            .mapValues(lambda conf: multi_vehicle_utils.gen_param(conf, 'throttle'))
            # PairRDD(vehicle, (data, throttle_param))
            .join(throttle_features)
            # RDD(plot_file)
            .map(lambda vehicle_data: multi_vehicle_plot_utils.gen_plot(vehicle_data, origin_prefix, 'throttle')))

        brake_features = spark_helper.cache_and_log(
            'brake_hdf5',
            origin_vehicle_dir
            # PairRDD(vehicle, abs_path_to_hdf5)
            .mapValues(lambda path: os.path.join(path, 'brake_calibration_table.pb.txt.hdf5'))
            # PairRDD(vehicle, data)
            .mapValues(read_h5))

        brake_plots = spark_helper.cache_and_log(
            'brake_plot',
            # PairRDD(vehicle, VEHICLE_PARAM_CONF.vehicle_param)
            vehicle_param_conf
            # PairRDD(vehicle, brake_param)
            .mapValues(lambda conf: multi_vehicle_utils.gen_param(conf, 'brake'))
            # PairRDD(vehicle, (data, brake_param))
            .join(brake_features)
            # RDD(plot_file)
            .map(lambda vehicle_data: multi_vehicle_plot_utils.gen_plot(vehicle_data, origin_prefix, 'brake')))


if __name__ == '__main__':
    MultiCalibrationTableVisualization().main()
