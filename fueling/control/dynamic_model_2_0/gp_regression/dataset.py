#!/usr/bin/env python

"""Extracting and processing dataset"""
import glob
import os

from scipy.signal import savgol_filter
from torch.utils.data.dataset import Dataset
import h5py
import numpy as np
import torch

from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import input_index
import fueling.common.logging as logging

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_LENGTH = 100
INPUT_DIM = feature_config["input_dim"]
OUTPUT_DIM = feature_config["output_dim"]
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]


class GPDataSet(Dataset):

    def __init__(self, args):
        """
        Initialization
        """
        self.training_data_path = args.training_data_path
        self.testing_data_path = args.testing_data_path
        self.use_cuda = args.use_cuda

    def get_train_data(self):
        """
        Generate training data from a list of labeled data
        """
        datasets = glob.glob(os.path.join(self.training_data_path, '**/*.h5'), recursive=True)
        logging.info(len(datasets))
        if self.use_cuda:
            input_data = torch.zeros(0, INPUT_LENGTH, INPUT_DIM).cuda
            output_data = torch.zeros(OUTPUT_DIM, 0).cuda
        else:
            input_data = torch.zeros(0, INPUT_LENGTH, INPUT_DIM)
            output_data = torch.zeros(OUTPUT_DIM, 0)
        for h5_file in datasets:
            logging.debug(os.path.join(h5_file))
            with h5py.File(h5_file, 'r') as model_norms_file:
                # Get input data
                input_segment = np.array(model_norms_file.get('input_segment'))
                # Smoothing noisy acceleration data
                input_segment[:, input_index["a"]] = savgol_filter(
                    input_segment[:, input_index["a"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
                input_segment = torch.from_numpy(input_segment)
                input_segment = input_segment.view(1, INPUT_LENGTH, INPUT_DIM)
                # Get output data
                output_segment = torch.tensor(np.array(model_norms_file.get('output_segment')))
                output_segment = output_segment.view(OUTPUT_DIM, 1)
                # Stack the data segments
                input_data = torch.cat((input_data, input_segment.float()), 0)
                output_data = torch.cat((output_data, output_segment.float()), 1)
        return (input_data, output_data)

    def get_test_data(self):
        """
        Generate testing data from a list of labeled data
        """
        datasets = glob.glob(os.path.join(self.testing_data_path, '**/*.h5'), recursive=True)
        logging.info(os.path.join(self.testing_data_path, '**/*.h5'))
        logging.info(len(datasets))
        input_data = torch.zeros(0, INPUT_LENGTH, INPUT_DIM)
        gt_data = torch.zeros(0, OUTPUT_DIM)
        for h5_file in datasets:
            logging.debug(os.path.join(h5_file))
            with h5py.File(h5_file, 'r') as model_norms_file:
                # Get input data
                input_segment = np.array(model_norms_file.get('input_segment'))
                # Smoothing noisy acceleration data
                input_segment[:, input_index["a"]] = savgol_filter(
                    input_segment[:, input_index["a"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
                input_segment = torch.from_numpy(input_segment)
                input_segment = input_segment.view(1, INPUT_LENGTH, INPUT_DIM)
                # Get output data
                gt_res_error = torch.tensor(np.array(model_norms_file.get('output_segment')))
                gt_res_error = gt_res_error.view(1, OUTPUT_DIM)
                # Stack the data segments
                input_data = torch.cat((input_data, input_segment.float()), 0)
                gt_data = torch.cat((gt_data, gt_res_error.float()), 0)
        return (input_data, gt_data)
