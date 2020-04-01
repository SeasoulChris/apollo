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
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


def train(args, dataset, gp_class):
    """Train the model"""
    features, labels = dataset.get_train_data()
    labels = labels.view(labels.shape[1], -1)
    # [window_size, batch_size, channel]
    features = torch.transpose(features, 0, 1)
    input_dim = features.shape[-1]
    output_dim = labels.shape[-1]
    batch_size = features.shape[-2]
    logging.info("************Input Dim: {}".format(features.shape))
    logging.info("************Output Dim: {}".format(labels.shape))

    # noise_prior
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)

    # Define the inducing points of Gaussian Process
    inducing_points = features[:, torch.arange(0, batch_size,
                                               step=int(max(batch_size / args.num_inducing_point, 1))).long(), :]
    logging.info('inducing points data shape: {}'.format(inducing_points.shape))
    encoder_net_model = Encoder(u_dim=input_dim, kernel_dim=args.kernel_dim)
    model = GPModel(inducing_points=inducing_points,
                    encoder_net_model=encoder_net_model, num_tasks=output_dim)
    likelihood.train()
    model.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    logging.info("Start of training")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.shape[-1])
    logging.info(labels.shape[0])
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(features)
        loss = -mll(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss.sum()))
        if epoch == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)

    test_features, test_labels = dataset.get_test_data()
    test_labels = labels.view(test_labels.shape[1], -1)
    test_features = torch.transpose(test_features, 0, 1)
    save_gp(model, test_features)
    return model


class MeanVarModelWrapper(nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def save_gp(model, test_x):
    wrapped_model = MeanVarModelWrapper(model)
    with gpytorch.settings.trace_mode(), torch.no_grad():
        fake_input = test_x
        pred = wrapped_model(fake_input)  # Compute caches
        traced_model = torch.jit.trace(wrapped_model, fake_input, check_trace=False)
        logging.info("saving model")
    traced_model.save('/tmp/traced_gp.pt')


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
