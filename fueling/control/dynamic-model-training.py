#!/usr/bin/env python

import glob

import h5py
import numpy as np

from scipy.signal import savgol_filter

import fueling.control.training_models.mlp_keras as mlp_keras
import fueling.control.training_models.lstm_keras as lstm_keras
from fueling.control.features.parameters_training import dim
from fueling.common.base_pipeline import BasePipeline

# Constants
DIM_INPUT = dim["pose"] + dim["incremental"] + \
    dim["control"]  # accounts for mps
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
        h5s = glob.glob(
            '/apollo/modules/data/fuel/fueling/control/data/hdf5/*/*/*.hdf5')
        dirs = '/apollo/modules/data/fuel/fueling/control/data/'
        data = (
            # h5
            self.get_spark_context().parallelize(h5s)
            # -> [segment]
            .map(lambda h5: self.generate_segment(h5))
            # -> dict
            .flatMap(lambda segments: self.load_data(segments))
            # -> dict, with unique keys.
            .reduceByKey(lambda rdd_1, rdd_2: np.vstack((rdd_1, rdd_2)))
            .cache())
        param_norm = (
            data.filter(lambda key_value: key_value[0] == 'mlp_input_data')
            # -> (mlp_input_data, param_norm)
            .mapValues(lambda rdd: self.get_param_norm(rdd))
            .values()
            .collect()[0])
        data.foreach(lambda rdd: mlp_keras.mlp_keras(
            rdd[0][1], rdd[1][1], param_norm, dirs))
        data.foreach(lambda rdd: lstm_keras.lstm_keras(
            rdd[2][1], rdd[3][1], param_norm, dirs))

    def run_prod(self):
        hdf5 = glob.glob(
            '/mnt/bos/modules/control/feature_extraction_hf5/hdf5_training/transit_2019/*/*/*.hdf5')
        dirs = '/mnt/bos/modules/control/'
        data = (
            self.get_spark_context().parallelize(hdf5)
            .map(lambda h5: self.generate_segment(h5))
            .flatMap(lambda segments: self.load_data(segments))
            .reduceByKey(np.vstack)
            .cache())
        param_norm = (
            data.map(lambda rdd: self.get_param_norm(rdd))
            .cache())
        data.foreach(lambda rdd: mlp_keras.mlp_keras(
            rdd[0][1], rdd[1][1], param_norm, dirs))
        data.foreach(lambda rdd: lstm_keras.lstm_keras(
            rdd[2][1], rdd[3][1], param_norm, dirs))

    def generate_segment(self, h5):
        # print('h5 files:', h5)
        segments = []
        print('################Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as fin:
            for ds in fin.itervalues():
                if len(segments) == 0:
                    segments.append(np.array(ds))
                else:
                    segments[-1] = np.concatenate((segments[-1],
                                                   np.array(ds)), axis=0)
        print('Segments: ', segments)
        return segments

    def get_param_norm(self, feature):
        """
        normalize the samples and save normalized parameters
        """
        #feature = rdd["mlp_input_data"]
        fea_mean = np.mean(feature, axis=0)
        fea_std = np.std(feature, axis=0) + 1e-6
        param_norm = (fea_mean, fea_std)
        return param_norm

    def load_data(self, segments):
        total_len = 0
        total_sequence_num = 0
        for segment in segments:
            total_len += (segment.shape[0] - TIME_STEPS)
            total_sequence_num += (segment.shape[0] -
                                   TIME_STEPS - DIM_LSTM_LENGTH)
        print('total length:', total_len)
        mlp_input_data = np.zeros([total_len, DIM_INPUT])
        mlp_output_data = np.zeros([total_len, DIM_OUTPUT])
        lstm_input_data = np.zeros(
            [total_sequence_num, DIM_INPUT, DIM_LSTM_LENGTH])
        lstm_output_data = np.zeros([total_sequence_num, DIM_OUTPUT])
        i = 0
        m = 0
        for segment in segments:
            # smooth IMU acceleration data
            # window size 51, polynomial order 3
            segment[:, 8] = savgol_filter(segment[:, 8], 51, 3)
            # window size 51, polynomial order 3
            segment[:, 9] = savgol_filter(segment[:, 9], 51, 3)

            for k in range(segment.shape[0]):
                if k >= TIME_STEPS:
                    mlp_input_data[i, 0] = segment[k -
                                                   TIME_STEPS, 14]  # speed mps
                    mlp_input_data[i, 1] = segment[k-TIME_STEPS, 8] * \
                        np.cos(segment[k-TIME_STEPS, 0]) + segment[k-TIME_STEPS, 9] * \
                        np.sin(segment[k-TIME_STEPS, 0])  # acc
                    mlp_input_data[i, 2] = segment[k -
                                                   TIME_STEPS, 13]  # angular speed
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
            for k in range(mlp_input_data.shape[0]):
                if k >= DIM_LSTM_LENGTH:
                    lstm_input_data[m, :, :] = np.transpose(
                        mlp_input_data[(k-DIM_LSTM_LENGTH):k, :])
                    lstm_output_data[m, :] = mlp_output_data[k, :]
                    m += 1
        training_data = [
            ("mlp_input_data", mlp_input_data),
            ("mlp_output_data", mlp_output_data),
            ("lstm_input_data", lstm_input_data),
            ("lstm_output_data", lstm_output_data)
        ]
        return training_data

    def concatenate_data(self, dict_1, dict_2):
        return {
            "mlp_input_data": np.vstack((dict_1["mlp_input_data"], dict_2["mlp_input_data"])),
            "mlp_output_data": np.vstack((dict_1["mlp_output_data"], dict_2["mlp_output_data"])),
            "lstm_input_data": np.vstack((dict_1["lstm_input_data"], dict_2["lstm_input_data"])),
            "lstm_output_data": np.vstack((dict_1["lstm_output_data"], dict_2["lstm_output_data"]))
        }


if __name__ == '__main__':
    DynamicModelTraining().run_test()
