#!/usr/bin/env python

import os

import colored_glog as glog
import h5py
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client
import fueling.control.dynamic_model.data_generator.data_generator as data_generator
import fueling.control.dynamic_model.model_factory.lstm_keras as lstm_keras
import fueling.control.dynamic_model.model_factory.mlp_keras as mlp_keras


class DynamicModelTraining(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        data_dir = '/apollo/modules/data/fuel/testdata/control/learning_based_model'
        output_dir = os.path.join(data_dir, 'dynamic_model_output')
        training_dataset = [os.path.join(data_dir, 'hdf5_training/training_test.hdf5')]
        # RDD(file_path) for training dataset.
        training_dataset_rdd = self.to_rdd(training_dataset)
        self.run(training_dataset_rdd, output_dir)

    def run_prod(self):
        prefix = 'modules/control/learning_based_model/hdf5_training/Mkz7/UniformDistributed'
        # RDD(file_path) for training dataset
        training_dataset_rdd = self.to_rdd(self.bos().list_files(prefix, '.hdf5'))
        output_dir = bos_client.abs_path('modules/control/learning_based_model/dynamic_model_output/')
        self.run(training_dataset_rdd, output_dir)

    def run(self, training_dataset_rdd, output_dir):
        data = (
            # RDD(absolute_file_path)
            training_dataset_rdd
            # RDD(training_data_segment)
            .map(data_generator.generate_segment)
            # RDD(training_data_segment), which is valid.
            .filter(lambda segment: segment is not None)
            # RDD(training_data_segment), smoothing input features.
            .map(data_generator.feature_preprocessing)
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
            .mapValues(lambda input_output: data_generator.get_param_norm(input_output[0], 
                                                                            input_output[1]))
            # param_norm
            .values()
            .first())
        glog.info('Param Norm = {}'.format(param_norm))

        def _train(data_item):
            key, (input_data, output_data) = data_item
            if key == 'mlp_data':
                mlp_keras.mlp_keras(input_data, output_data, param_norm, output_dir)
            elif key == 'lstm_data':
                lstm_keras.lstm_keras(input_data, output_data, param_norm, output_dir)
        data.foreach(_train)


if __name__ == '__main__':
    DynamicModelTraining().main()
