#!/usr/bin/env python

import os

import glob
import h5py
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.logging as logging
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.dynamic_model.data_generator.training_data_generator as data_generator
import fueling.control.dynamic_model.model_factory.lstm_keras as lstm_keras
import fueling.control.dynamic_model.model_factory.mlp_keras as mlp_keras

VEHICLE_ID = feature_config["vehicle_id"]
IS_BACKWARD = feature_config["is_backward"]


class DynamicModelTraining(BasePipeline):

    def run_test(self):
        data_dir = '/apollo/modules/data/fuel/testdata/control/learning_based_model'
        output_dir = os.path.join(data_dir, 'dynamic_model_output')
        if IS_BACKWARD:
            training_dataset = glob.glob(
                os.path.join(data_dir, 'hdf5_training/Mkz7/UniformDistributed/backward/*/*/*.hdf5'))
        else:
            training_dataset = glob.glob(
                os.path.join(data_dir, 'hdf5_training/Mkz7/UniformDistributed/forward/*/*/*.hdf5'))
        # RDD(file_path) for training dataset.
        training_dataset_rdd = self.to_rdd(training_dataset)
        self.run(training_dataset_rdd, output_dir)

    def run_prod(self):
        dataset_dir = 'modules/control/learning_based_model/hdf5_training'
        if IS_BACKWARD:
            prefix = os.path.join(dataset_dir, VEHICLE_ID, 'UniformDistributed/backward')
        else:
            prefix = os.path.join(dataset_dir, VEHICLE_ID, 'UniformDistributed/forward')
        # RDD(file_path) for training dataset
        training_dataset_rdd = self.to_rdd(self.our_storage().list_files(prefix, '.hdf5'))
        output_dir = self.our_storage().abs_path(
            'modules/control/learning_based_model/dynamic_model_output/')
        self.run(training_dataset_rdd, output_dir)

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
