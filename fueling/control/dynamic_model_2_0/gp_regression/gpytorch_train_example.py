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
import pyro
import pickle

from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gp_model_example import GPModelExample
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
import fueling.common.logging as logging


def encoding(dataset):
    feature, label = dataset.get_train_data()
    logging.info("************Input Dim: {}".format(feature.shape))
    logging.info("************Output Dim: {}".format(label.shape))
    encoder = Encoder(args, feature.shape[2], args.kernel_dim)
    encoded_feature = encoder.forward(feature)
    logging.info("************Input Dim: {}".format(encoded_feature.shape))


def get_dataset():
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
    return train_x, train_y, train_loader, test_x, test_y


def train_gp(train_x, train_y, train_loader):
    """Train the model"""
    inducing_points = train_x[:500, :]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(variance=0.1 * torch.ones(2, 1))
    model = GPModelExample(inducing_points=inducing_points, input_data_dim=train_x.shape[0])
    likelihood.train()
    model.train()

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
        if i == 2:
            gpytorch.settings.tridiagonal_jitter(1e-4)
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # save and load model
    state_dict = model.state_dict()
    logging.info(state_dict)
    torch.save(model.state_dict(),
               '/fuel/fueling/control/dynamic_model_2_0/gp_regression/traced_gp_example.pth')
    state_dict = torch.load(
        '/fuel/fueling/control/dynamic_model_2_0/gp_regression/traced_gp_example.pth')
    model = GPModelExample(inducing_points, train_x.shape[0])
    model.load_state_dict(state_dict)
    logging.info(model.state_dict)
    return model, likelihood


class MeanVarModelWrapper(nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def save_gp(model, test_x):
    wrapped_model = MeanVarModelWrapper(model)
    with torch.no_grad():
        fake_input = test_x
        pred = wrapped_model(fake_input)  # Compute caches
        traced_model = torch.jit.trace(wrapped_model, fake_input, check_trace=False)
        logging.info("saving model")
    traced_model.save('/fuel/fueling/control/dynamic_model_2_0/gp_regression/traced_gp_example.pt')


if __name__ == '__main__':
    train_x, train_y, train_loader, test_x, test_y = get_dataset()
    gp_model, gp_likelihood = train_gp(train_x, train_y, train_loader)

    # prediction & evaluation
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    gp_model.eval()
    gp_likelihood.eval()
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = gp_model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]
    print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))

    save_gp(gp_model, test_x)

# parser = argparse.ArgumentParser(description='GP')
# # paths
# parser.add_argument(
#     '--training_data_path',
#     type=str,
#     default="/fuel/fueling/control/dynamic_model_2_0/testdata/training")
# parser.add_argument(
#     '--testing_data_path',
#     type=str,
#     default="/fuel/fueling/control/dynamic_model_2_0/testdata/test_dataset")
# parser.add_argument(
#     '--gp_model_path',
#     type=str,
#     default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model")
# parser.add_argument('--kernel_dim', type=int, default=20)
# args = parser.parse_args()
# dataset = GPDataSet(args)
