#!/usr/bin/env python
from math import floor
import argparse
import os
import time

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as Func

from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
import fueling.common.logging as logging


def train(args, dataset, gp_class):
    """Train the model"""
    feature, label = dataset.get_train_data()
    # get data
    logging.info("************Input Dim: {}".format(feature.shape))
    logging.info("************Output Dim: {}".format(label.shape))
    logging.info("************Output Example: {}".format(label[0]))

    # noise_prior
    likelihood = gpytorch.likelihoods.GaussianLikelihood(variance=0.1 * torch.ones(2, 1))

    # Define the inducing points of Gaussian Process
    inducing_points = feature[torch.arange(0, feature.shape[0],
                                           step=int(max(feature.shape[0] / args.num_inducing_point, 1))).long()]
    logging.info('inducing points data shape: {}'.format(inducing_points.shape))
    logging.info('feature data shape: {}'.format(feature.shape))
    model = GPModel(inducing_points=inducing_points, input_data_dim=feature.shape[-1])

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    logging.info("Start of training")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=label.shape[0])

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model.forward(feature)
        loss = -mll(output, label)
        loss.sum().backward(retain_graph=True)
        optimizer.step()
        logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss.sum()))
        if epoch == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)

    # save_gp(args, gp_instante, feature, encoder)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/training")
    # default = "/fuel/fueling/control/dynamic_model_2_0/testdata/labeled_data"
    parser.add_argument(
        '--testing_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/test_dataset")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/results")
    # parser.add_argument(
    #     '--online_gp_model_path',
    #     type=str,
    #     default="/fuel/fueling/control/dynamic_model_2_0/testdata/20191004-130454")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argment to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    args = parser.parse_args()
    dataset = GPDataSet(args)
    train(args, dataset, GPModel)
