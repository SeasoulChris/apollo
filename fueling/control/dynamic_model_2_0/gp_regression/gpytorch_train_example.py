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


def encoding(dataset):
    feature, label = dataset.get_train_data()
    logging.info("************Input Dim: {}".format(feature.shape))
    logging.info("************Output Dim: {}".format(label.shape))
    encoder = Encoder(args, feature.shape[2], args.kernel_dim)
    encoded_feature = encoder.forward(feature)
    logging.info("************Input Dim: {}".format(encoded_feature.shape))


def get_dataset(dataset):
    X, y = torch.randn(1000, 3), torch.randn(1000)
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    return train_x, train_y, train_loader


def train_gp(train_x, train_y, train_loader):
    inducing_points = train_x[:500, :]
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model.train()
    likelihood.train()

    # We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    num_epochs = 10

    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(i, loss))
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train_x, train_y, train_loader = get_dataset()
    train_gp(train_x, train_y, train_loader)

    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/training")
    parser.add_argument(
        '--testing_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/test_dataset")
    parser.add_argument('--kernel_dim', type=int, default=20)
    args = parser.parse_args()
    dataset = GPDataSet(args)
    encoding(dataset)
