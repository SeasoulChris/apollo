#!/usr/bin/env python
import argparse
import math
import os

import numpy as np
import torch

from fueling.learning.train_utils import cuda
from fueling.planning.datasets.img_in_traj_out_dataset import ImgInTrajOutDataset
from fueling.planning.models.trajectory_imitation_model import TrajectoryImitationModel


def calculate_displacement_error(pred, y):
    y_true = y.view(y.size(0), -1)
    out = pred - y_true
    # 30 points with 5 properties x,y,phi,v, a
    out.view(30, 5)
    pos_x_diff = out[:, 0]
    pos_y_diff = out[:, 1]
    phi_diff = out[:, 2]
    v_diff = out[:, 3]
    a_diff = out[:, 4]
    displacement_error = torch.mean(
        torch.sqrt(pos_x_diff ** 2 + pos_y_diff ** 2)).item()
    heading_error = torch.mean(phi_diff).item()
    v_error = torch.mean(v_diff).item()
    a_error = torch.mean(a_diff).item()
    return displacement_error, heading_error, v_error, a_error


def model_evaluator(test_loader, model):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []
        a_errors = []
        for i, (X, y) in enumerate(test_loader):
            X, y = cuda(X), cuda(y)
            pred = model(X)
            displacement_error, heading_error, v_error, a_error = calculate_displacement_error(
                pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            a_errors.append(a_error)

        average_displacement_error = np.mean(displcement_errors)
        average_heading_error = np.mean(heading_errors)
        average_v_error = np.mean(v_errors)
        average_a_error = np.mean(a_errors)
        print('average displacement error: {}.'.format(
            average_displacement_error))
        print('average heading error: {}.'.format(average_heading_error))
        print('average speed error: {}.'.format(average_v_error))
        print('average acceleration error: {}.'.format(average_a_error))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('model_file', type=str, help='trained model')
    parser.add_argument('test_set_folder', type=str, help='test data')
    args = parser.parse_args()

    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_dataset = ImgInTrajOutDataset(args.test_set_folder)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                              num_workers=4, drop_last=True)

    model = TrajectoryImitationModel()
    model_state_dict = torch.load(args.model_file)
    model.load_state_dict(model_state_dict)

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    model_evaluator(test_loader, model)
