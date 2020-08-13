#!/usr/bin/env python

"""Extracting and processing dataset"""

import pickle
import os

from torch.utils.data import Dataset
import h5py
import numpy as np
import torch

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.proto.fnn_model_pb2 import GPModelParam

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_DIM = feature_config["input_dim"]
OUTPUT_DIM = feature_config["output_dim"]
INPUT_WINDOW_SIZE = feature_config["input_window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]
PI = 3.14159


def get_standardization_factors(datasets):
    """Compute mean and variance of input data using only training data"""
    # mean
    input_segment_mean = np.zeros((1, INPUT_DIM))
    for dataset in datasets:
        input_segment = dataset[0]  # input segment (100, 6)
        input_segment_mean += np.mean(input_segment, axis=0)
    input_segment_mean = input_segment_mean / len(datasets)

    # standard deviation
    input_segment_std = np.zeros((1, INPUT_DIM))
    for dataset in datasets:
        input_segment = dataset[0]  # dataset[0] : input
        value = (input_segment - input_segment_mean) ** 2
        input_segment_std += np.mean(value, axis=0)
    input_segment_std = np.sqrt(input_segment_std / len(datasets))
    return (input_segment_mean, input_segment_std)


def save_model_param_to_bin(input_mean, input_std, dst_dir):
    """ dump params to bin file for on-line interface """
    gp_model_param = GPModelParam()
    gp_model_param.input_dim = INPUT_DIM
    gp_model_param.output_dim = OUTPUT_DIM
    gp_model_param.input_window_size = INPUT_WINDOW_SIZE
    gp_model_param.standardization_factor.input_mean.columns.extend(input_mean.reshape(-1).tolist())
    gp_model_param.standardization_factor.input_std.columns.extend(input_std.reshape(-1).tolist())

    # export proto to bin
    param_bin_file = os.path.join(dst_dir, 'param.bin')
    proto_utils.write_pb_to_bin_file(gp_model_param, param_bin_file)

    # export proto to txt (for debug)
    param_txt_file = os.path.join(os.path.dirname(dst_dir), 'param.txt')
    proto_utils.write_pb_to_text_file(gp_model_param, param_txt_file)


def extract_data_from_file(h5_file):
    """ run in bos; assume there is no NAN files"""
    logging.log_every_n(logging.INFO, f'loading {h5_file}', 500)
    with h5py.File(h5_file, 'r') as labeled_data_file:
        # Get input data
        input_segment = np.array(labeled_data_file.get('input_segment'))
        output_segment = np.array(labeled_data_file.get('output_segment'))
        return (input_segment, output_segment)


class BosSetDataset(Dataset):
    def __init__(self, storage, input_data_dir, validation_set, is_train=True, param_file=None):
        super().__init__()
        self.dst_dir = input_data_dir
        self.is_train = is_train
        self.files = storage.list_files(input_data_dir, suffix='.h5')
        self.dataset_file_list = []
        self.get_datasets(validation_set)
        # load standardization factors
        params = proto_utils.get_pb_from_bin_file(param_file, GPModelParam())
        self.input_mean = params.standardization_factor.input_mean.columns
        self.input_std = params.standardization_factor.input_std.columns
        logging.info(f'param factors are {self.input_mean} and {self.input_std}')

    def get_datasets(self, validation_set):
        # loop over list files
        for cur_file in self.files:
            if self.is_train == (cur_file not in validation_set):
                # logging.info(cur_file)
                self.dataset_file_list.append(cur_file)

    def __len__(self):
        return len(self.dataset_file_list)

    def __getitem__(self, idx):
        # extract data set
        cur_dataset = extract_data_from_file(self.dataset_file_list[idx])
        # standardization
        standardized_input = (cur_dataset[0] - self.input_mean) / self.input_std
        return (torch.from_numpy(standardized_input).float(),
                torch.from_numpy(cur_dataset[1]).float())

    def get_len(self):
        return self.__len__()


class BosDataset(Dataset):
    """ Dateset for BOS """

    def __init__(self, file_list_dir, is_train=True, param_file=None):
        super().__init__()
        self.file_list_dir = file_list_dir
        self.datasets = []
        self.get_datasets()
        if is_train and (not param_file):
            # generate standardization factors
            self.input_mean, self.input_std = get_standardization_factors(self.datasets)
            save_model_param_to_bin(self.input_mean, self.input_std, self.file_list_dir)
        else:
            # load standardization factors
            params = proto_utils.get_pb_from_bin_file(param_file, GPModelParam())
            self.input_mean = params.standardization_factor.input_mean.columns
            self.input_std = params.standardization_factor.input_std.columns
            logging.info(f'param factors are {self.input_mean} and {self.input_std}')

    def get_datasets(self):
        # load the file from file list
        files = file_utils.list_files_with_suffix(self.file_list_dir, '.txt')
        # loop over list files
        for cur_file in files:
            logging.info(cur_file)
            with open(cur_file, "rb") as fp:  # Pickling
                data_list = pickle.load(fp)
                # loop over data list
                for cur_data_file in data_list:
                    logging.info(cur_data_file)
                    self.extract_data_from_file(cur_data_file)

    def extract_data_from_file(self, h5_file):
        """ run in bos; assume there is no NAN files"""
        with h5py.File(h5_file, 'r') as labeled_data_file:
            # Get input data
            input_segment = np.array(labeled_data_file.get('input_segment'))
            output_segment = np.array(labeled_data_file.get('output_segment'))
            self.datasets.append((input_segment, output_segment))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        standardized_input = (self.datasets[idx][0] - self.input_mean) / self.input_std
        return (torch.from_numpy(standardized_input).float(),
                torch.from_numpy(self.datasets[idx][1]).float())

    def get_len(self):
        return self.__len__()


class DynamicModelDataset(Dataset):
    """ data preparation for dynamic model """

    def __init__(self, data_dir, is_train=True, model_dir=None, factor_file=None,
                 is_normalize=False, is_standardize=True):
        super().__init__()
        self.data_dir = data_dir
        self.is_normalize = is_normalize
        self.is_standardize = is_standardize

        if not model_dir:
            model_dir = '/fuel/fueling/control/dynamic_model_2_0/testdata/mlp_model/forward'

        self.model_dir = model_dir
        self.get_pre_normalization_factors()

        self.get_datasets()

        if is_train and (not factor_file):
            self.standardization_factors = dict()
            self.normalization_factors = dict()
            self.standardization_factors_file = os.path.join(
                self.data_dir, 'standardization_factors.npy')
            self.normalization_factors_file = os.path.join(
                self.data_dir, 'normalization_factors.npy')
            self.set_standardization_factors()
            self.save_standardization_factors_to_bin()
            self.set_normalization_factors()
        else:
            # for validation and test data, use same normalization factors as training data set.
            # for processed training data (factors already saved) load factor directly
            # factor_file path is training data path
            self.standardization_factors_file = os.path.join(
                factor_file, 'standardization_factors.npy')
            self.normalization_factors_file = os.path.join(
                factor_file, 'normalization_factors.npy')
            # load training data factors
            self.standardization_factors = np.load(
                self.standardization_factors_file, allow_pickle=True).item()
            self.normalization_factors = np.load(
                self.normalization_factors_file, allow_pickle=True).item()
            logging.info(
                f'loading normalization factors from {self.normalization_factors_file}'
                + f'as {self.normalization_factors}')

    def save_standardization_factors_to_bin(self):
        """ dump params to bin file for on-line interface """
        gp_model_param = GPModelParam()
        gp_model_param.input_dim = INPUT_DIM
        gp_model_param.output_dim = OUTPUT_DIM
        gp_model_param.input_window_size = INPUT_WINDOW_SIZE
        gp_model_param.standardization_factor.input_mean.columns.extend(
            self.standardization_factors['mean'].reshape(-1).tolist())
        gp_model_param.standardization_factor.input_std.columns.extend(
            self.standardization_factors['std'].reshape(-1).tolist())
        standardization_factors_bin_file = os.path.join(
            self.data_dir, 'standardization_factors.bin')
        with open(standardization_factors_bin_file, 'wb') as bin_file:
            bin_file.write(gp_model_param.SerializeToString())
        # export proto to txt
        txt_file_name = os.path.join(self.data_dir, 'standardization_factors.txt')
        # export single frame to txt for debug
        proto_utils.write_pb_to_text_file(gp_model_param, txt_file_name)

    def get_pre_normalization_factors(self):
        """ if the model is pre-normalized, get the normalization factor"""
        model_norms_path = os.path.join(self.model_dir, 'norms.h5')
        with h5py.File(model_norms_path, 'r') as model_norms_file:
            self.pre_input_mean = np.array(model_norms_file.get('input_mean'))
            self.pre_input_std = np.array(model_norms_file.get('input_std'))

    def get_datasets_w_pre_denormalization(self):
        """Extract datasets from data path"""
        # list of dataset = (input_tensor, output_tensor)
        self.datasets = []
        h5_files = file_utils.list_files_with_suffix(self.data_dir, '.h5')
        for idx, h5_file in enumerate(h5_files):
            logging.debug(f'h5_file: {h5_file}')
            with h5py.File(h5_file, 'r') as model_norms_file:
                # Get input data
                input_segment = np.array(model_norms_file.get('input_segment'))
                if np.isnan(np.sum(input_segment)):
                    raise Exception(f'file {h5_file} contains NAN data in input segment')
                # denormalize input data from DM10
                input_segment[:, range(0, 5)] = input_segment[:, range(0, 5)] * \
                    self.pre_input_std + self.pre_input_mean
                # heading angle covert back to PI
                input_segment[:, -1] = input_segment[:, -1] * PI
                # Get output data
                output_segment = np.array(model_norms_file.get('output_segment'))
                if np.isnan(np.sum(output_segment)):
                    raise Exception(f'file {h5_file} contains NAN data in output segment')
                self.datasets.append((input_segment, output_segment))

    def get_datasets(self):
        """ when input data for DM1.0 is not normalized """
        self.datasets = []
        h5_files = file_utils.list_files_with_suffix(self.data_dir, '.h5')
        for idx, h5_file in enumerate(h5_files):
            logging.debug(f'h5_file: {h5_file}')
            with h5py.File(h5_file, 'r') as model_norms_file:
                # Get input data
                input_segment = np.array(model_norms_file.get('input_segment'))
                if np.isnan(np.sum(input_segment)):
                    raise Exception(f'file {h5_file} contains NAN data in input segment')
                output_segment = np.array(model_norms_file.get('output_segment'))
                if np.isnan(np.sum(output_segment)):
                    raise Exception(f'file {h5_file} contains NAN data in output segment')
                self.datasets.append((input_segment, output_segment))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        processed_inputs = self.pre_process(self.datasets[idx][0])
        return (torch.from_numpy(processed_inputs).float(),
                torch.from_numpy(self.datasets[idx][1]).float())

    def get_len(self):
        return self.__len__()

    def pre_process(self, input_data):
        """ pre processing, standardize or normalize """
        if self.is_standardize:
            return self.standardize(input_data)
        if self.is_normalize:
            return self.normalize(input_data)
        return input_data

    def getitem(self, idx):
        """ for debuging """
        return self.__getitem__(idx)

    def set_normalization_factors(self):
        """Compute min and max value of input data for each feature"""
        input_max = np.full((1, INPUT_DIM), np.NINF)
        input_min = np.full((1, INPUT_DIM), np.Inf)
        for i, dataset in enumerate(self.datasets):
            input_max = np.maximum(input_max, np.amax(dataset[0], axis=0))
            input_min = np.minimum(input_min, np.amin(dataset[0], axis=0))
        # save min and max to npy file
        self.normalization_factors['max'] = input_max
        self.normalization_factors['min'] = input_min
        np.save(self.normalization_factors_file, self.normalization_factors)
        return input_max, input_min

    def set_standardization_factors(self):
        """Compute mean and variance of input data using only training data"""
        # mean
        logging.info(len(self.datasets))
        input_segment_mean = np.zeros((1, INPUT_DIM))
        for i, dataset in enumerate(self.datasets):
            input_segment = dataset[0]  # input segment (100, 6)
            input_segment_mean += np.mean(input_segment, axis=0)
        input_segment_mean = input_segment_mean / len(self.datasets)

        # standard deviation
        input_segment_std = np.zeros((1, INPUT_DIM))
        for i, dataset in enumerate(self.datasets):
            input_segment = dataset[0]  # dataset[0] : input
            value = (input_segment - input_segment_mean) ** 2
            input_segment_std += np.mean(value, axis=0)
        input_segment_std = np.sqrt(input_segment_std / len(self.datasets))

        # save mean and standard deviation to npy file
        self.standardization_factors['mean'] = input_segment_mean
        self.standardization_factors['std'] = input_segment_std
        np.save(self.standardization_factors_file, self.standardization_factors)
        return input_segment_mean, input_segment_std

    def standardize(self, inputs):
        """Standardize given data"""
        input_mean = self.standardization_factors['mean']
        input_std = self.standardization_factors['std']
        inputs_standardized = (inputs - input_mean) / input_std
        return inputs_standardized

    def normalize(self, inputs):
        """Normalize given data"""
        input_max = self.normalization_factors['max']
        input_min = self.normalization_factors['min']
        inputs_normalized = (inputs - input_min) / (input_max - input_min)
        return inputs_normalized
