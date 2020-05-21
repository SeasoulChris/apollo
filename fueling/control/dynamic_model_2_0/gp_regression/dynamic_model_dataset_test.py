#!/usr/bin/env python

import os


import matplotlib.pyplot as plt
import numpy as np
import torch


from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import DynamicModelDataset
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging

if __name__ == '__main__':
    dynamic_model_dataset = DynamicModelDataset(
        '/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/train')
    logging.info(f'dataset length {len(dynamic_model_dataset.datasets)}')
    processed_data = dynamic_model_dataset.getitem(0)[0]
    for id in range(1, len(dynamic_model_dataset.datasets)):
        processed_data = torch.cat((processed_data, dynamic_model_dataset.getitem(id)[0]), 0)
    logging.info(f'processed data shape is {processed_data.shape}')
    # visualize standardized data
    for id in range(0, 6):
        plt.figure(figsize=(12, 8))
        plt.plot(processed_data[:, id], 'b.')
        logging.info(
            f'mean value for {id} is {np.mean(processed_data[:, id].numpy(), dtype=np.float64)}')
        logging.info(
            f'std value for {id} is {np.std(processed_data[:, id].numpy(), dtype=np.float64)}')
        logging.info(
            f'max value for {id} is {np.amax(processed_data[:, id].numpy())}')
        logging.info(
            f'min value for {id} is {np.amin(processed_data[:, id].numpy())}')
        plt.show()
