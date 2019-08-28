#!/usr/bin/env python

"""Extracting and processing dataset"""
import glob
import os
import sys

from torch.utils.data.dataset import Dataset
import colored_glog as glog
import h5py
import numpy as np
import torch

from fueling.control.dynamic_model.gp_regression.model_conf import segment_index, feature_config
from fueling.control.dynamic_model.gp_regression.model_conf import input_index, output_index

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_LENGTH = 100
DIM_INPUT = feature_config["input_dim"]
DIM_OUTPUT = feature_config["output_dim"]


class GPDataSet(Dataset):

    def __init__(self, args):
        """
        Initialization
        """
        self.data_path = args.labeled_data_path

    def get_train_data(self):
        """
        Generate training data from a list of labeled data
        """
        datasets = glob.glob(os.path.join(self.data_path, '*.hdf5'))
        input_data = torch.zeros(0, INPUT_LENGTH, DIM_INPUT)
        output_data = torch.zeros(DIM_OUTPUT, 0)
        for h5_file in datasets:
            with h5py.File(h5_file, 'r') as model_norms_file:
                input_segment = torch.tensor(np.array(model_norms_file.get('input_segment')))
                output_segment = torch.tensor(np.array(model_norms_file.get('output_segment')))
                input_segment = input_segment.view(1, INPUT_LENGTH, DIM_INPUT)
                output_segment = output_segment.view(DIM_OUTPUT, 1)
                input_data = torch.cat((input_data, input_segment), 0)
                output_data = torch.cat((output_data, output_segment), 1)
        return (input_data, output_data)
