#!/usr/bin/env python

import os

import glob
import h5py
import shutil
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.dynamic_model.data_generator.training_data_generator as data_generator
import fueling.control.dynamic_model.flag
import fueling.control.dynamic_model.model_factory.lstm_keras as lstm_keras
import fueling.control.dynamic_model.model_factory.mlp_keras as mlp_keras


MODEL_CONF = 'model_config.py'


class DynamicModelTraining(BasePipeline):

    def run_test(self):
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        IS_BACKWARD = self.FLAGS.get('is_backward')
        data_dir = '/fuel/testdata/control/generated_uniform'
        if IS_BACKWARD:
            training_data_path = os.path.join(data_dir, job_owner, 'backward', job_id)
        else:
            training_data_path = os.path.join(data_dir, job_owner, 'forward', job_id)
        model_dir = '/fuel/testdata/control/learning_based_model'
        output_dir = os.path.join(model_dir, 'dynamic_model_output')
        model_conf_prefix = '/fuel/fueling/control/dynamic_model/conf'

        vehicles = multi_vehicle_utils.get_vehicle(training_data_path)
        logging.info('vehicles = {}'.format(vehicles))
        # run test as a vehicle ID
        for vehicle in vehicles:
            self.load_model_conf(vehicle, model_conf_prefix, IS_BACKWARD)
            self.execute_task(vehicle, model_conf_prefix, training_data_path, output_dir)

    def run_prod(self):
        # intermediate result folder
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        IS_BACKWARD = self.FLAGS.get('is_backward')
        our_storage = self.our_storage()
        data_dir = 'modules/control/tmp/uniform'

        if IS_BACKWARD:
            data_prefix = os.path.join(data_dir, job_owner, 'backward', job_id)
        else:
            data_prefix = os.path.join(data_dir, job_owner, 'forward', job_id)

        training_data_path = our_storage.abs_path(data_prefix)
        output_dir = bos_client.abs_path(
            'modules/control/learning_based_model/dynamic_model_output/')
        # TODO: V2.
        model_conf_prefix = '/fuel/fueling/control/dynamic_model/conf'
        # get vehicles
        vehicles = multi_vehicle_utils.get_vehicle(training_data_path)
        logging.info('vehicles = {}'.format(vehicles))
        # run proc as a vehicle ID
        for vehicle in vehicles:
            self.load_model_conf(vehicle, model_conf_prefix, IS_BACKWARD)
            self.execute_task(vehicle, model_conf_prefix, training_data_path, output_dir)

    def load_model_conf(self, vehicle, model_conf_prefix, is_backward):
        if is_backward:
            # load backward model_conf for vehicle
            model_conf_target_prefix = os.path.join(model_conf_prefix, vehicle, 'backward')
        else:
            # load farward model_conf for vehicle
            model_conf_target_prefix = os.path.join(model_conf_prefix, vehicle)
        file_utils.makedirs(model_conf_prefix)
        shutil.copyfile(os.path.join(model_conf_target_prefix, MODEL_CONF),
                        os.path.join(model_conf_prefix, MODEL_CONF))
        logging.info('model_conf_target_prefix: %s' % model_conf_target_prefix)

    def execute_task(self, vehicle, model_conf_prefix, training_data_path, output_dir):
        # vehicle dir
        vehicle_dir = os.path.join(training_data_path, vehicle)
        # model output dir
        model_output_dir = os.path.join(output_dir, vehicle)
        logging.info('vehicle_dir = {}'.format(vehicle_dir))
        logging.info('model_output_dir = {}'.format(model_output_dir))
        # RDD hd5_dataset
        hd5_files_path = glob.glob(os.path.join(vehicle_dir, '*/*.hdf5'))
        # logging.info('hd5_files_path = {}'.format(hd5_files_path))
        # for file in files_path:
        training_dataset_rdd = self.to_rdd(hd5_files_path)
        self.run(training_dataset_rdd, model_output_dir)

    def run(self, training_dataset_rdd, output_dir):
        data = (
            # RDD(absolute_file_path)
            training_dataset_rdd
            # RDD(training_data_segment)
            .map(feature_extraction.generate_segment)
            # RDD(training_data_segment), which is valid.
            .filter(lambda segment: segment is not None)
            # RDD(training_data_segment), smoothing input features.
            .map(feature_extraction.feature_preprocessing)
            # RDD(training_data_segment), which is valid after feature_preprocessing.
            .filter(lambda segment: segment is not None)
            # RDD('mlp_data|lstm_data', (input, output)).
            .flatMap(data_generator.generate_training_data)
            # RDD('mlp_data|lstm_data', (input, output)), which is valid.
            .filter(lambda data: data is not None)
            # RDD('mlp_data|lstm_data', (input, output)), with unique keys.
            .reduceByKey(lambda data_1, data_2: (np.concatenate((data_1[0], data_2[0]), axis=0),
                                                 np.concatenate((data_1[1], data_2[1]), axis=0)))
            .cache())

        param_norm = (
            # RDD('mlp_data', (input, output))
            data.filter(lambda key_value: key_value[0] == 'mlp_data')
            # RDD('mlp_data', param_norm)
            .mapValues(lambda input_output: feature_extraction.get_param_norm(input_output[0],
                                                                              input_output[1]))
            # param_norm
            .values()
            .first())
        logging.info('Param Norm = {}'.format(param_norm))

        def _train(data_item):
            key, (input_data, output_data) = data_item
            if key == 'mlp_data':
                mlp_keras.mlp_keras(input_data, output_data, param_norm, output_dir)
            elif key == 'lstm_data':
                lstm_keras.lstm_keras(input_data, output_data, param_norm, output_dir)

        data.foreach(_train)

if __name__ == '__main__':
    DynamicModelTraining().main()
