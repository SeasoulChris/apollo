#!/usr/bin/env python
"""Training models"""

import os
import time

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
import fueling.common.logging as logging

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
        self.conv1 = nn.Conv1d(u_dim, 200, u_dim, stride=3)  # 32
        self.conv2 = nn.Conv1d(200, 100, u_dim, stride=3)  # 9
        self.conv3 = nn.Conv1d(100, 100, u_dim, stride=3)  # 2
        self.fc = nn.Linear(200, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        logging.debug("Original data shape: {}".format(data.shape))
        conv1_input = torch.transpose(data, -1, -2)
        logging.debug("Conv1 input data shape: {}".format(conv1_input.shape))
        conv2_input = Func.relu(self.conv1(conv1_input))
        logging.debug("Conv2 input data shape: {}".format(conv2_input.shape))
        conv3_input = Func.relu(self.conv2(conv2_input))
        logging.debug("Conv3 input data shape: {}".format(conv3_input.shape))
        fc_input = Func.relu(self.conv3(conv3_input))
        logging.debug("Fully-connected layer input data shape: {}".format(fc_input.shape))
        data = self.fc(fc_input.view(fc_input.shape[0], -1))
        return data


def preprocessing(args, dataset, gp):
    # TODO(Jiaxuan): Implement data preprocessing module
    logging.info("End of preprocessing")


def save_gp(args, gp_model, kernel_net):
    """Save the learned models for Gaussian process"""
    timestr = time.strftime('%Y%m%d-%H%M%S')
    model_saving_path = os.path.join(args.gp_model_path, timestr)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)
    torch.save(gp_model.gp_f.state_dict(), os.path.join(model_saving_path, "gp_f.p"))
    torch.save(gp_model.gp_f.kernel.state_dict(), os.path.join(model_saving_path, "kernel.p"))
    torch.save(gp_model.gp_f.likelihood.state_dict(),
               os.path.join(model_saving_path, "likelihood.p"))
    torch.save(kernel_net.state_dict(), os.path.join(model_saving_path, "fnet.p"))


def train_gp(args, dataset, gp_class):
    """Train the dataset with Gaussian Process assumption"""
    feature, label = dataset.get_train_data()
    logging.info("************Input Dim: {}".format(feature.shape))
    logging.info("************Input Example: {}".format(feature[0]))
    logging.info("************Output Dim: {}".format(label.shape))
    logging.info("************Output Example: {}".format(label[0]))

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
                              step=int(max(feature.shape[0] / args.num_inducing_point, 1))).long()]
    # The Pyro core of Gaussian Process training through variational inference
    gp_f = gp.models.VariationalSparseGP(feature, label, kernel, Xu,
                                         num_data=feature.shape[0], likelihood=likelihood,
                                         mean_function=None, whiten=True, jitter=1e-3)
    # TODO(Jiaxuan): Define the cooresponding GP class
    # Instantiate a Gaussian Process object
    gp_instante = gp_class(args, gp_f, dataset)
    # args.mate = preprocessing(args, dataset, gp_instante)
    # Pyro Adam opitmizer
    optimizer = optim.ClippedAdam({"lr": args.lr, "lrd": args.lr_decay})
    # svi: Stochastic variational inference
    # ELBO: Evidence Lower Boundary
    svi = infer.SVI(gp_instante.model, gp_instante.guide, optimizer, infer.Trace_ELBO())

    logging.info("Start of training")
    gp_instante.set_data(feature, label)
    for epoch in range(1, args.epochs + 1):
        loss = svi.step()
        logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss))
        if epoch == 10:
            gp_instante.gp_f.jitter = 1e-4

    save_gp(args, gp_instante, deep_encoding_net)
