#!/usr/bin/env python

import torch
import torch.nn as nn

import fueling.common.logging as logging


class GPEncoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        logging.info('u_dim: {}'.format(u_dim))
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
        self.conv1 = nn.Conv1d(u_dim, 200, u_dim, stride=3)  # 5
        self.conv2 = nn.Conv1d(200, 100, u_dim, stride=3)  # 4
        self.conv3 = nn.Conv1d(100, 100, u_dim, stride=9)
        self.relu = nn.ReLU()

    def forward(self, data):
        """Define forward computation and activation functions"""
        logging.info("Original data shape: {}".format(data.shape))
        conv_input = torch.transpose(data, -1, -2).detach()
        # 100*10*6 => 10
        logging.info("Conv1 input data shape: {}".format(conv_input.shape))
        conv1_input = conv_input.detach()
        tmp = self.conv1(conv1_input).detach()
        conv2_input = self.relu(tmp)
        logging.info("Conv2 input data shape: {}".format(conv2_input.shape))
        conv3_input = self.relu(self.conv2(conv2_input))
        logging.info("Conv3 input data shape: {}".format(conv3_input.shape))
        fc_input = self.relu(self.conv3(conv3_input))
        logging.info("Fully-connected layer input data shape: {}".format(fc_input.shape))
        # logging.info(fc_input.view(fc_input.shape[0], -1))
        data = self.fc(fc_input.view(fc_input.shape[0], -1))  # 200 *
        # fc_input = self.conv_net(conv_input)
        # logging.info(fc_input.view(fc_input.shape[0], -1))
        # data = self.fc(fc_input.view(fc_input.shape[0], -1))
        logging.info(fc_input.shape[0])
        logging.info("Encoded data shape: {}".format(data.shape))
        return data
