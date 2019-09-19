#!/usr/bin/env python
from collections import Counter
import glob
import operator
import os

from absl import flags
import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.control.common.training_conf import inter_result_folder
from fueling.control.common.training_conf import output_folder
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.control.common.multi_job_utils as multi_job_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.features.calibration_table_train_utils as train_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as CalibrationTable


def get_data_from_hdf5(hdf5_rdd):
    return (
        # PairRDD('throttle or brake', list of hdf5 files)
        hdf5_rdd
        # PairRDD('throttle or brake', segments)
        .mapValues(train_utils.generate_segments)
        # PairRDD('throttle or brake', data)
        .mapValues(train_utils.generate_data))


class MultiJobTrain(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'multi_job_training')

    def get_feature_hdf5_prod(self, feature_dir, throttle_or_brake, train_or_test):
        return (
            # PairRDD(vehicle, feature folder)
            feature_dir
            # PairRDD(vehicle, throttle/brake train/test folder)
            .mapValues(lambda feature_dir: os.path.join(feature_dir,
                                                        throttle_or_brake, train_or_test))
            # PairRDD(vehicle, all files in throttle/brake train/test folder)
            .flatMapValues(lambda path: self.bos().list_files(path, '.hdf5'))
            # PairRDD((vehicle, 'throttle or brake'), hdf5 files)
            .map(lambda (vehicle, hdf5_file): ((vehicle, throttle_or_brake), hdf5_file))
            # PairRDD((vehicle, 'throttle or brake'), hdf5 files RDD)
            .groupByKey()
            # PairRDD((vehicle, 'throttle or brake'), list of hdf5 files)
            .mapValues(list))

    def run_test(self):
        """Run test."""
        # target folder is the same as origin folder for test case
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated_conf'

        def get_feature_hdf5_files(feature_dir, throttle_or_brake, train_or_test):
            return (
                # PairRDD(vehicle, feature folder)
                feature_dir
                # PairRDD(vehicle, throttle/brake train/test feature folder)
                .mapValues(lambda feature_dir:
                           os.path.join(feature_dir, throttle_or_brake, train_or_test))
                # PairRDD(vehicle, all files in throttle train feature folder)
                .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*.hdf5')))
                # PairRDD((vehicle, 'throttle or brake'), hdf5 files)
                .map(lambda (vehicle, hdf5_file): ((vehicle, throttle_or_brake), hdf5_file))
                # PairRDD((vehicle, 'throttle or brake'), hdf5 files RDD)
                .groupByKey()
                # PairRDD((vehicle, 'throttle or brake'), list of hdf5 files)
                .mapValues(list))

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type)))

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_train_files = spark_helper.cache_and_log(
            'throttle_train_files',
            get_feature_hdf5_files(origin_vehicle_dir, 'throttle', 'train'))

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_test_files = get_feature_hdf5_files(origin_vehicle_dir, 'throttle', 'test')

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_train_files = get_feature_hdf5_files(origin_vehicle_dir, 'brake', 'train')

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_test_files = get_feature_hdf5_files(origin_vehicle_dir, 'brake', 'test')

        feature_dir = (throttle_train_files, throttle_test_files,
                       brake_train_files, brake_test_files)

        self.run(feature_dir, origin_vehicle_dir, target_prefix)

    def run_prod(self):
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        bos_client = BosClient()

        # intermediate result folder
        origin_prefix = os.path.join(inter_result_folder, job_owner, job_id)
        origin_dir = bos_client.abs_path(origin_prefix)

        # output folder
        target_prefix = os.path.join(output_folder, job_owner, job_id)
        target_dir = bos_client.abs_path(target_prefix)

        # use dir (abs_path) to get conf files
        origin_vehicle_conf_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_dir])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle: os.path.join(origin_dir, vehicle)))

        # use prefix to list files
        # RDD(origin_dir)
        origin_vehicle_data_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_dir])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_train_files = spark_helper.cache_and_log(
            'throttle_train_files',
            self.get_feature_hdf5_prod(origin_vehicle_data_dir, 'throttle', 'train'))

        # PairRDD((vehicle, 'throttle'), list of hdf5 files)
        throttle_test_files = spark_helper.cache_and_log(
            'throttle_test_files',
            self.get_feature_hdf5_prod(origin_vehicle_data_dir, 'throttle', 'test'))

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_train_files = spark_helper.cache_and_log(
            'brake_train_files',
            self.get_feature_hdf5_prod(origin_vehicle_data_dir, 'brake', 'train'))

        # PairRDD((vehicle, 'brake'), list of hdf5 files)
        brake_test_files = spark_helper.cache_and_log(
            'brake_test_files',
            self.get_feature_hdf5_prod(origin_vehicle_data_dir, 'brake', 'test'))

        feature_dir = (throttle_train_files, throttle_test_files,
                       brake_train_files, brake_test_files)

        self.run(feature_dir, origin_vehicle_conf_dir, target_dir)

    def run(self, feature_dir, origin_vehicle_conf_dir, target_dir):

        # conf files
        logging.info('origin_vehicle_dir %s' % origin_vehicle_conf_dir.collect())
        # # # get confs
        # PairRDD(vehicle, vehicle_conf)
        vehicle_conf = origin_vehicle_conf_dir.mapValues(multi_vehicle_utils.get_vehicle_param)
        # PairRDD(vehicle, train_conf)
        train_conf = origin_vehicle_conf_dir.mapValues(multi_job_utils.get_train_conf)

        # PairRDD(vehicle, (vehicle_conf, train_conf))
        conf = vehicle_conf.join(train_conf)

        throttle_train_files, throttle_test_files, brake_train_files, brake_test_files = feature_dir

        # PairRDD((vehicle, 'throttle'), train data)
        throttle_train_data = spark_helper.cache_and_log(
            'throttle_train_data',
            get_data_from_hdf5(throttle_train_files))
        # PairRDD((vehicle, 'throttle'), test data)
        throttle_test_data = spark_helper.cache_and_log(
            'throttle_test_data',
            get_data_from_hdf5(throttle_test_files))

        throttle_data = spark_helper.cache_and_log(
            'throttle_data',
            # PairRDD((vehicle, 'throttle'), (x_train_data, y_train_data))
            throttle_train_data
            # PairRDD((vehicle, 'throttle'),
            #         ((x_train_data, y_train_data), (x_test_data, y_test_data)))
            .join(throttle_test_data))

        throttle_train_param = spark_helper.cache_and_log(
            'throttle_train_param',
            # PairRDD(vehicle, train_param)
            conf.mapValues(lambda (vehicle_conf, train_conf):
                           multi_vehicle_utils.gen_param_w_train_conf(vehicle_conf, train_conf,
                                                                      'throttle'))
            # PairRDD((vehicle, 'throttle'), train_param)
            .map(lambda (vehicle, train_param): ((vehicle, 'throttle'), train_param)))

        throttle_model = spark_helper.cache_and_log(
            'throttle_model',
            # PairRDD((vehicle, 'throttle'), (data_set, train_param))
            throttle_data.join(throttle_train_param)
            # PairRDD(vehicle, table_filename)
            .map(lambda elem: train_utils.train_write_model(elem, target_dir)))

        """ brake """
        # PairRDD((vehicle, 'brake'), train data)
        brake_train_data = spark_helper.cache_and_log(
            'brake_train_data',
            get_data_from_hdf5(brake_train_files))
        # PairRDD((vehicle, 'brake'), test data)
        brake_test_data = spark_helper.cache_and_log(
            'brake_test_data',
            get_data_from_hdf5(brake_test_files))

        brake_data = spark_helper.cache_and_log(
            'brake_data',
            # PairRDD((vehicle, 'brake'), (x_train_data, y_train_data))
            brake_train_data
            # PairRDD((vehicle, 'brake'),
            #         ((x_train_data, y_train_data), (x_test_data, y_test_data)))
            .join(brake_test_data))

        brake_train_param = spark_helper.cache_and_log(
            'brake_train_param',
            # PairRDD(vehicle, train_param)
            conf.mapValues(lambda (vehicle_conf, train_conf):
                           multi_vehicle_utils.gen_param_w_train_conf(vehicle_conf, train_conf,
                                                                      'brake'))
            # PairRDD((vehicle, 'brake'), train_param)
            .map(lambda (vehicle, train_param): ((vehicle, 'brake'), train_param)))

        brake_model = spark_helper.cache_and_log(
            'brake_model',
            # PairRDD((vehicle, 'brake'), (data_set, train_param))
            brake_data.join(brake_train_param)
            # PairRDD(vehicle, table_filename)
            .map(lambda elem: train_utils.train_write_model(elem, target_dir)))
        print('brake_model.collect() ', brake_model.collect())
        print('throttle_model.collect() ', throttle_model.collect())

        model = spark_helper.cache_and_log(
            'model',
            brake_model.join(throttle_model).mapValues(train_utils.combine_file))


if __name__ == '__main__':
    MultiJobTrain().main()
