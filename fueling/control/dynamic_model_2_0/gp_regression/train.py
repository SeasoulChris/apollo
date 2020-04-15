#!/usr/bin/env python
import argparse
import os
import time

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gpytorch
import torch
import torch.nn as nn

from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


def train_and_save(args, dataset, gp_class):
    """Train the model"""
    features, labels = dataset.get_train_data()
    # [window_size, batch_size, channel]
    features = torch.transpose(features, 0, 1)
    # [batch_size, channel] (window_size = 1)
    labels = torch.transpose(labels, 0, 1)
    input_dim = features.shape[-1]
    output_dim = labels.shape[-1]
    batch_size = features.shape[-2]
    logging.info("************Input Dim: {}".format(features.shape))
    logging.info("************Output Dim: {}".format(labels.shape))

    # noise_prior
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)

    # Define the inducing points of Gaussian Process
    # inducing_points = features[:, torch.arange(0, batch_size,
    #                                            step=int(max(batch_size / args.num_inducing_point, 1))).long(), :]
    step_size = int(max(batch_size / args.num_inducing_point, 1))
    logging.info(f'step size is: {step_size}')
    inducing_point_num = torch.arange(0, batch_size, step=step_size).long()
    logging.info(f'inducing point indices are {inducing_point_num}')
    inducing_points = features[:, inducing_point_num, :]
    logging.info('inducing points data shape: {}'.format(inducing_points.shape))
    encoder_net_model = Encoder(u_dim=input_dim, kernel_dim=args.kernel_dim)
    model = GPModel(inducing_points=inducing_points,
                    encoder_net_model=encoder_net_model, num_tasks=output_dim)
    if args.use_cuda:
        likelihood.train().cuda
        model.train().cuda
    else:
        likelihood.train()
        model.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=args.lr)
    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    logging.info("Start of training")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.shape[-1])
    logging.info(labels.shape[0])
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(features)
        loss = -mll(output, labels)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss.sum()))
        if epoch == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)
    # return

    # save model as state_dict
    timestr = time.strftime('%Y%m%d-%H%M%S')
    # with time stamp
    state_dict_file_path = os.path.join(args.gp_model_path, timestr)
    if not os.path.exists(state_dict_file_path):
        os.makedirs(state_dict_file_path)
    save_model_state_dict(model, likelihood, state_dict_file_path)

    # save model as torchscript
    # with time stamp
    jit_file_path = os.path.join(args.online_gp_model_path, timestr)
    test_features, test_labels = dataset.get_test_data()
    test_features = torch.transpose(test_features, 0, 1)
    if not os.path.exists(jit_file_path):
        os.makedirs(jit_file_path)
    save_model_torch_script(model, test_features, jit_file_path)


class MeanVarModelWrapper(nn.Module):
    def __init__(self, gp):
        super().__init__()
        self.gp = gp

    def forward(self, x):
        output_dist = self.gp(x)
        return output_dist.mean, output_dist.variance


def save_model_torch_script(model, test_features, jit_file_path):
    '''save to TorchScript'''
    file_name = os.path.join(jit_file_path, "gp_online.pt")
    wrapped_model = MeanVarModelWrapper(model)
    with gpytorch.settings.trace_mode(), torch.no_grad():
        pred = wrapped_model(test_features)  # Compute cache
        traced_model = torch.jit.trace(wrapped_model, test_features, check_trace=False)
        logging.info(f'saving model: {file_name}')
    traced_model.save(file_name)


def save_model_state_dict(model, likelihood, state_dict_file_path):
    '''save as state_dict'''
    model.eval()
    likelihood.eval()
    file_name = os.path.join(state_dict_file_path, "gp.pth")
    state_dict = model.state_dict()
    logging.info(f'saving model state dict: {file_name}')
    torch.save(model.state_dict(), file_name)


if __name__ == '__main__':

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
    # offline model
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/evaluation_result")
    # on-line model
    parser.add_argument(
        '--online_gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argment to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    # argument to use cuda or not for training
    parser.add_argument('--use_cuda', type=bool, default=False)
    args = parser.parse_args()
    dataset = GPDataSet(args)
    train_and_save(args, dataset, GPModel)
