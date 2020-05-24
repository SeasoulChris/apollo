#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import fueling.common.logging as logging


class Encoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, stride=4)
        self.fc = nn.Linear(250, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        logging.debug(data[0, -1, :])
        # original data shape: [sequency/window_size, batch_size, channel]
        # conv_input shape: [batch_size, channel, sequency/window_size]
        logging.debug(torch.transpose(data, 1, 0).shape)
        conv1_input = torch.transpose(torch.transpose(data, 1, 0), -2, -1)
        logging.debug(conv1_input.shape)
        data = F.relu(self.conv1(conv1_input))
        logging.debug(data.shape)
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        logging.debug(data.shape)
        return data
