#!/usr/bin/env python

"""Training models"""

import os
import time

import colored_glog as glog
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
import pyro.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as Func

from fueling.control.dynamic_model.gp_regression.model_conf import segment_index, feature_config
from fueling.control.dynamic_model.gp_regression.model_conf import input_index, output_index

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_LENGTH = feature_config["DELTA_T"] / feature_config["delta_t"]  # Default 100


class DeepEncodingNet(nn.Module):
    """Convolutional neural network to encode high-dimentional features"""

    def __init__(self, args, u_dim, kernel_dim):
        """Network initialization"""
        super(DeepEncodingNet, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=5)
        self.conv2 = nn.Conv1d(100, 50, u_dim, stride=5)
        self.fc = nn.Linear(200, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        data = Func.relu(self.conv2(Func.relu(self.conv1(torch.transpose(data, -1, -2)))))
        data = self.fc(data.view(data.shape[0], -1))
        return data


def preprocessing(args, dataset, gp):
    # TODO(Jiaxuan): Implement data preprocessing module
    glog.info("End of preprocessing")


def save_gp(args, gp_model, kernel_net):
    # TODO(Jiaxuan): Implement model saving module
    glog.info("Model saved")


def train_gp(args, dataset, gp_class):
    """Train the dataset with Gaussian Process assumption"""
    feature, label = dataset.get_train_data()
    glog.info("************Input Dim: {}".format(feature.shape))
    glog.info("************Input Example: {}".format(feature[0]))
    glog.info("************Output Dim: {}".format(label.shape))
    glog.info("************Output Example: {}".format(label[0]))

    # Encode the original features to lower-dimentional feature with kernal size
    deep_encoding_net = DeepEncodingNet(args, feature.shape[2], args.kernel_dim)

    def _encoded_feature(original_feature):
        return pyro.module("DeepEncodingNet", deep_encoding_net)(original_feature)

    likelihood = gp.likelihoods.Gaussian(variance=0.1 * torch.ones(2, 1))
    kernelization = gp.kernels.Matern52(input_dim=args.kernel_dim,
                                        lengthscale=torch.ones(args.kernel_dim))
    # Kernalize the encoding features at lower-dimension
    kernel = gp.kernels.Warping(kernelization, iwarping_fn=_encoded_feature)
    # Define the inducing points of Gaussian Process
    Xu = feature[torch.arange(0, feature.shape[0],
                              step=int(feature.shape[0] / args.num_inducing_point)).long()]
    # The Pyro core of Gaussian Process training through variational inference
    gp_model = gp.models.VariationalSparseGP(feature, label, kernel, Xu,
                                             num_data=feature.shape[0], likelihood=likelihood,
                                             mean_function=None, whiten=True, jitter=1e-3)
    # TODO(Jiaxuan): Define the cooresponding GP class
    # Instantiate a Gaussian Process object
    gp_instante = gp_class(args, gp_model, dataset)
    # args.mate = preprocessing(args, dataset, gp_instante)
    # Pyro Adam opitmizer
    optimizer = optim.ClippedAdam({"lr": args.lr, "lrd": args.lr_decay})
    # svi: Stochastic variational inference
    # ELBO: Evidence Lower Boundary
    svi = infer.SVI(gp_instante.model, gp_instante.guide, optimizer, infer.Trace_ELBO())

    glog.info("Start of training")
    gp_instante.set_data(feature, label)
    for epoch in range(1, args.epochs + 1):
        loss = svi.step()
        glog.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss))
        if epoch == 10:
            gp_instante.gp_f.jitter = 1e-4
        save_gp(args, gp_instante, deep_encoding_net)
