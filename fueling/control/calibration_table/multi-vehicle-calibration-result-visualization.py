#!/usr/bin/env python

import glob
import os
import time

from absl import flags
from matplotlib.backends.backend_pdf import PdfPages
import colored_glog as glog
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.common.h5_utils import read_h5
from fueling.common.partners import partners
from fueling.control.common.training_conf import inter_result_folder
from fueling.control.common.training_conf import output_folder
import fueling.common.bos_client as bos_client
import fueling.common.email_utils as email_utils
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
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')

        # conf file in the input data folder
        conf_prefix = os.path.join(
            inter_result_folder, job_owner, job_id, 'CalibrationTableFeature')

        # results in output folder
        # origin_prefix = self.FLAGS.get('output_data_path')
        origin_prefix = os.path.join(output_folder, job_owner, job_id)

        origin_dir = bos_client.abs_path(origin_prefix)
        conf_dir = bos_client.abs_path(conf_prefix)

        # RDD(plot_file)
        plot_files = self.run(origin_dir, conf_dir)

        partner = partners.get(job_owner)
        if partner and partner.email:
            title = 'Your vehicle calibration job is done!'
            content = 'Please find the result in attachments.'
            receivers = [partner.email, 'apollo_internal@baidu.com']
            # TODO: Add the generated calibration table to the attachments
            attachments = plot_files.collect()
            email_utils.send_email_info(title, content, receivers, attachments)

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
            # PairRDD(vehicle_in_the_list, vehicle)
            # .filter(lambda (vehicle, _): vehicle in vehicle_list)
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

        vehicle_param_conf = spark_helper.cache_and_log(
            'conf_file',
            # RDD(abs_path_to_folder)
            self.to_rdd([conf_prefix])
            # RDD(vehicle)
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle, vehicle)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_in_the_list, vehicle)
            # .filter(lambda (vehicle, _): vehicle in vehicle_list)
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
            vehicle_param_conf
            # PairRDD(vehicle, brake_param)
            .mapValues(lambda conf: multi_vehicle_utils.gen_param(conf, 'brake'))
            # PairRDD(vehicle, (data, brake_param))
            .join(brake_features)
            # RDD(plot_file)
            .map(lambda vehicle_data:
                 multi_vehicle_plot_utils.gen_plot(vehicle_data, origin_prefix, 'brake')))
        # RDD(plot_file)
        return throttle_plots.union(brake_plots)


if __name__ == '__main__':
    MultiCalibrationTableVisualization().main()
