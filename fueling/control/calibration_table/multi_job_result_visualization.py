#!/usr/bin/env python

import glob
import os
import time

from absl import flags
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.common.h5_utils import read_h5
from fueling.common.partners import partners
from fueling.control.common.training_conf import inter_result_folder
from fueling.control.common.training_conf import output_folder
import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.control.common.multi_job_utils as multi_job_utils
import fueling.control.common.multi_vehicle_plot_utils as multi_vehicle_plot_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils


class MultiJobResultVisualization(BasePipeline):

    def run_test(self):
        origin_dir = '/fuel/testdata/control/generated_conf'
        self.run(origin_dir, origin_dir)

    def run_prod(self):
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')

        # conf file in the input data folder
        conf_prefix = os.path.join(inter_result_folder, job_owner, job_id)

        # results in output folder
        origin_prefix = os.path.join(output_folder, job_owner, job_id)

        our_storage = self.our_storage()
        origin_dir = our_storage.abs_path(origin_prefix)
        conf_dir = our_storage.abs_path(conf_prefix)

        # RDD(plot_file)
        plot_files = self.run(origin_dir, conf_dir)


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
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        # PairRDD((vehicle, 'throttle'), hdf5_file)
        throttle_features = spark_helper.cache_and_log(
            'throttle_hdf5',
            origin_vehicle_dir
            # PairRDD(vehicle, abs_path_to_hdf5)
            .mapValues(lambda path: os.path.join(path, 'throttle_calibration_table.pb.txt.hdf5'))
            # PairRDD(vehicle, data)
            .mapValues(read_h5))

        conf_files = spark_helper.cache_and_log(
            'conf_file',
            # RDD(abs_path_to_folder)
            self.to_rdd([conf_prefix])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle, abs_path_to_vehicle_folder)
            .mapValues(lambda vehicle: os.path.join(conf_prefix, vehicle)))

        # PairRDD(vehicle, VEHICLE_PARAM_CONF.vehicle_param)
        vehicle_conf = conf_files.mapValues(multi_vehicle_utils.get_vehicle_param)
        # PairRDD(vehicle, train_conf)
        train_conf = conf_files.mapValues(multi_job_utils.get_train_conf)

        throttle_plots = spark_helper.cache_and_log(
            'throttle_plot',
            # PairRDD(vehicle, VEHICLE_PARAM_CONF.vehicle_param)
            vehicle_conf
            # PairRDD(vehicle, (vehicle_conf,train_conf)))
            .join(train_conf)
            # PairRDD(vehicle, throttle_param)
            .mapValues(lambda vehicle_train_conf:
                       multi_vehicle_utils.gen_param_w_train_conf(
                           vehicle_train_conf[0], vehicle_train_conf[1], 'throttle'))
            # PairRDD(vehicle, (data, throttle_param))
            .join(throttle_features)
            # RDD(plot_file)
            .map(lambda vehicle_data:
                 multi_vehicle_plot_utils.gen_plot(vehicle_data, origin_prefix, 'throttle')))

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
            vehicle_conf
            # PairRDD(vehicle, (vehicle_conf,train_conf)))
            .join(train_conf)
            # PairRDD(vehicle, brake_param)
            .mapValues(lambda vehicle_train_conf:
                       multi_vehicle_utils.gen_param_w_train_conf(
                           vehicle_train_conf[0], vehicle_train_conf[1], 'brake'))
            # PairRDD(vehicle, (data, brake_param))
            .join(brake_features)
            # RDD(plot_file)
            .map(lambda vehicle_data:
                 multi_vehicle_plot_utils.gen_plot(vehicle_data, origin_prefix, 'brake')))
        # RDD(plot_file)
        return throttle_plots.union(brake_plots)


if __name__ == '__main__':
    MultiJobResultVisualization().main()
