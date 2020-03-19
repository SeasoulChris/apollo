#!/usr/bin/env python

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional neural network to encode high-dimensional features"""

    def __init__(self, args, u_dim, kernel_dim):
        """Network initialization"""
        super().__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
        # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv_net = nn.Sequential(
            nn.Conv1d(u_dim, 200, u_dim, stride=3),  # 32
            nn.ReLU(),
            nn.Conv1d(200, 100, u_dim, stride=3),  # 9
            nn.ReLU(),
            nn.Conv1d(100, 100, u_dim, stride=3),  # 2
            nn.ReLU()
        )
        self.fc = nn.Linear(200, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # logging.debug("Original data shape: {}".format(data.shape))
        conv_input = torch.transpose(data, -1, -2).detach()
        # logging.debug("Conv1 input data shape: {}".format(conv_input.shape))
        # conv1_input = conv1_input.detach()
        # tmp = self.conv1(conv1_input).detach()
        # conv2_input = self.relu(tmp)
        # logging.debug("Conv2 input data shape: {}".format(conv2_input.shape))
        # conv3_input = self.relu(self.conv2(conv2_input))
        # logging.debug("Conv3 input data shape: {}".format(conv3_input.shape))
        # fc_input = self.relu(self.conv3(conv3_input))
        # logging.debug("Fully-connected layer input data shape: {}".format(fc_input.shape))
        fc_input = self.conv_net(conv_input)
        data = self.fc(fc_input.view(fc_input.shape[0], -1))
        return data
