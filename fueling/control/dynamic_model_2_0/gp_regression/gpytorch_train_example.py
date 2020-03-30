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
import pickle

from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gp_model_example import GPModelExample
import fueling.common.logging as logging


def get_dataset():
    ''' naive data sets'''
    X, y = torch.randn(1000, 3), torch.randn(1000)
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    logging.info(f'train_x size: {train_x.shape}')
    logging.info(f'train_dataset size: {train_dataset[0][0].shape}')
    logging.info(f'train_dataset size: {train_dataset[799][0].shape}')

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    return train_x, train_y, train_loader, test_x, test_y, test_loader


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
    return model, likelihood, inducing_points


def save_model(model, likelihood, file_path):
    '''save as state_dict'''
    model.eval()
    likelihood.eval()
    # save and load model
    state_dict = model.state_dict()
    logging.info(f'saving model state dict: {state_dict}')
    torch.save(model.state_dict(), file_path)


def load_model(inducing_points, train_x, file_path):
    '''load from state dict'''
    state_dict = torch.load(file_path)
    model = GPModelExample(inducing_points, train_x.shape[0])
    model.load_state_dict(state_dict)
    logging.info(f'loading model state dict: {model.state_dict}')
    return model


def predict(model, test_loader):
    '''predict model'''
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    return means[1:]


class MeanVarModelWrapper(nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def save_gp(model, test_x, jit_file_path):
    '''save to TorchScript'''
    wrapped_model = MeanVarModelWrapper(model)
    with gpytorch.settings.trace_mode(), torch.no_grad():
        pred = wrapped_model(test_x)  # Compute caches
        traced_model = torch.jit.trace(wrapped_model, test_x)
        logging.info('saving model')
    traced_model.save(jit_file_path)


if __name__ == '__main__':
    FILE_PATH = '/fuel/fueling/control/dynamic_model_2_0/gp_regression/traced_gp_example.pth'
    JIT_FILE_PATH = '/fuel/fueling/control/dynamic_model_2_0/gp_regression/traced_gp_example.pt'
    train_x, train_y, train_loader, test_x, test_y, test_loader = get_dataset()
    gp_model, gp_likelihood, inducing_point = train_gp(train_x, train_y, train_loader)
    save_model(gp_model, gp_likelihood, FILE_PATH)
    loaded_gp_model = load_model(inducing_point, train_x, FILE_PATH)
    means = predict(loaded_gp_model, test_loader)
    # evaluation
    logging.info('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y.cpu()))))
    save_gp(gp_model, test_x, JIT_FILE_PATH)
