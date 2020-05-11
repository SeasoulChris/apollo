#!/usr/bin/env python

from scipy.signal import savgol_filter
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import input_index, output_index


# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_LENGTH = 100
INPUT_DIM = feature_config["input_dim"]
OUTPUT_DIM = feature_config["output_dim"]
WINDOW_SIZE = feature_config["window_size"]
POLYNOMINAL_ORDER = feature_config["polynomial_order"]


class DynamicModelDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        # list of feature = (input_tensor, output_tensor)
        self.features = []

        h5_files = file_utils.list_files_with_suffix(data_dir, '.h5')
        for idx, h5_file in enumerate(h5_files):
            logging.debug(f'h5_file: {h5_file}')
            with h5py.File(h5_file, 'r') as model_norms_file:
                # Get input data
                input_segment = np.array(model_norms_file.get('input_segment'))
                # Smoothing noisy acceleration data
                input_segment[:, input_index["a"]] = savgol_filter(
                    input_segment[:, input_index["a"]], WINDOW_SIZE, POLYNOMINAL_ORDER)
                # Get output data
                output_segment = np.array(model_norms_file.get('output_segment'))
                self.features.append((input_segment, output_segment))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def getitem(self, idx):
        return self.__getitem__(idx)


if __name__ == '__main__':
    dynamic_model_dataset = DynamicModelDataset(
        '/fuel/fueling/control/dynamic_model_2_0/testdata/labeled_data')
    logging.info(f'dateset is {dynamic_model_dataset.getitem(10)}')
