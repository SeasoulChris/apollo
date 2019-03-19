#!/usr/bin/env python

import glob
import os

from scipy.signal import savgol_filter
import h5py
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.parameters_training import dim
import fueling.common.colored_glog as glog
import fueling.common.s3_utils as s3_utils
import fueling.control.dynamic_model.lstm_keras as lstm_keras
import fueling.control.dynamic_model.mlp_keras as mlp_keras


# Constants
DIM_INPUT = dim["pose"] + dim["incremental"] + dim["control"]  # accounts for mps
DIM_OUTPUT = dim["incremental"]  # the speed mps is also output
INPUT_FEATURES = ["speed mps", "speed incremental",
                  "angular incremental", "throttle", "brake", "steering"]
DIM_LSTM_LENGTH = dim["timesteps"]
TIME_STEPS = 1
EPOCHS = 10


class DynamicModelTraining(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        data_dir = '/apollo/modules/data/fuel/testdata/control/learning_based_model'
        output_dir = os.path.join(data_dir, 'dynamic_model_output')
        training_dataset = [os.path.join(data_dir, 'hdf5/training.hdf5')]
        # RDD(file_path) for training dataset
        training_dataset_rdd = self.get_spark_context().parallelize(training_dataset)
        self.run(training_dataset_rdd, output_dir)

    def run_prod(self):
        bucket = 'apollo-platform'
        prefix = 'modules/control/feature_extraction_hf5/hdf5_training/transit_2019'
        training_dataset_rdd = (
            # RDD(file_path) for training dataset, which starts with the prefix.
            s3_utils.list_files(bucket, prefix)
            # RDD(file_path) for training dataset, which ends with 'hdf5'
            .filter(lambda path: path.endswith('.hdf5'))
            # RDD(absolute_file_path)
            .map(s3_utils.abs_path))
        output_dir = s3_utils.abs_path('modules/control/dynamic_model_output/')
        self.run(training_dataset_rdd, output_dir)

    def run(self, training_dataset_rdd, output_dir):
        data = (
            # RDD(absolute_file_path)
            training_dataset_rdd
            # RDD(training_data_segment)
            .map(self.generate_segment)
            # RDD(training_data_segment), which is valid
            .filter(lambda segment: segment is not None)
            # RDD('mlp_data|lstm_data', (input, output))
            .flatMap(self.load_data)
            # RDD('mlp_data|lstm_data', (input, output)), with unique keys.
            .reduceByKey(lambda data_1, data_2: (np.vstack((data_1[0], data_2[0])),
                                                   np.vstack((data_1[1], data_2[1]))))
            .cache())

        param_norm = (
            # RDD('mlp_data', (input, output))
            data.filter(lambda key_value: key_value[0] == 'mlp_data')
            # RDD('mlp_data', param_norm)
            .mapValues(lambda input_output: self.get_param_norm(input_output[0]))
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

    def generate_segment(self, h5):
        # print('h5 files:', h5)
        segment = None
        glog.info('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as fin:
            for ds in fin.itervalues():
                if segment is None:
                    segment = np.array(ds)
                else:
                    segment = np.concatenate((segment, np.array(ds)), axis=0)
        return segment

    def get_param_norm(self, feature):
        """
        normalize the samples and save normalized parameters
        """
        #feature = rdd["mlp_input_data"]
        fea_mean = np.mean(feature, axis=0)
        fea_std = np.std(feature, axis=0) + 1e-6
        param_norm = (fea_mean, fea_std)
        return param_norm

    def load_data(self, segment):
        total_len = segment.shape[0] - TIME_STEPS
        total_sequence_num = segment.shape[0] - TIME_STEPS - DIM_LSTM_LENGTH
        glog.info('Total length: {}'.format(total_len))
        # smooth IMU acceleration data
        # window size 51, polynomial order 3
        segment[:, 8] = savgol_filter(segment[:, 8], 51, 3)
        # window size 51, polynomial order 3
        segment[:, 9] = savgol_filter(segment[:, 9], 51, 3)

        mlp_input_data = np.zeros([total_len, DIM_INPUT])
        mlp_output_data = np.zeros([total_len, DIM_OUTPUT])
        i = 0
        for k in range(TIME_STEPS, segment.shape[0]):
            mlp_input_data[i, 0] = segment[k - TIME_STEPS, 14]  # speed mps
            mlp_input_data[i, 1] = segment[k-TIME_STEPS, 8] * \
                np.cos(segment[k-TIME_STEPS, 0]) + segment[k-TIME_STEPS, 9] * \
                np.sin(segment[k-TIME_STEPS, 0])  # acc
            mlp_input_data[i, 2] = segment[k - TIME_STEPS, 13]  # angular speed
            # control from chassis
            mlp_input_data[i, 3] = segment[k-TIME_STEPS, 15]
            # control from chassis
            mlp_input_data[i, 4] = segment[k-TIME_STEPS, 16]
            # control from chassis
            mlp_input_data[i, 5] = segment[k-TIME_STEPS, 17]
            mlp_output_data[i, 0] = segment[k, 8] * \
                np.cos(segment[k, 0]) + segment[k, 9] * \
                np.sin(segment[k, 0])  # acc next
            # angular speed next
            mlp_output_data[i, 1] = segment[k, 13]
            i += 1

        lstm_input_data = np.zeros([total_sequence_num, DIM_INPUT, DIM_LSTM_LENGTH])
        lstm_output_data = np.zeros([total_sequence_num, DIM_OUTPUT])
        m = 0
        for k in range(DIM_LSTM_LENGTH, mlp_input_data.shape[0]):
            lstm_input_data[m, :, :] = np.transpose(mlp_input_data[(k-DIM_LSTM_LENGTH):k, :])
            lstm_output_data[m, :] = mlp_output_data[k, :]
            m += 1
        training_data = [
            ("mlp_data", (mlp_input_data, mlp_output_data)),
            ("lstm_data", (lstm_input_data, lstm_output_data))
        ]
        return training_data


if __name__ == '__main__':
    DynamicModelTraining().run_prod()
