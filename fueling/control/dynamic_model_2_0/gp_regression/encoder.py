#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import fueling.common.logging as logging


class Encoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        # super().__init__()
        # # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
        # # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.fc = nn.Linear(200, kernel_dim)
        # self.conv1 = nn.Conv1d(u_dim, 200, u_dim, stride=3)
        # self.conv2 = nn.Conv1d(200, 100, u_dim, stride=3)
        # self.conv3 = nn.Conv1d(100, 100, u_dim, stride=3)
        # self.tanh = nn.Tanh()

        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, stride=4)
        self.fc = nn.Linear(250, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # logging.info(data[0, -1, :])
        # # original data shape: [sequency/window_size, batch_size, channel]
        # logging.debug("Original data shape: {}".format(data.shape))
        # # conv_input shape: [batch_size, channel, sequency/window_size]
        logging.debug(torch.transpose(data, 1, 0).shape)
        conv1_input = torch.transpose(torch.transpose(data, 1, 0), -2, -1)
        logging.debug(conv1_input.shape)
        # conv2_input = F.relu(self.conv1(conv1_input))
        # conv3_input = F.relu(self.conv2(conv2_input))
        # fc_input = F.relu(self.conv3(conv3_input))
        # data = self.fc(fc_input.view(fc_input.shape[0], -1))
        # logging.info(data)
        # return data
        data = F.relu(self.conv1(conv1_input))
        logging.debug(data.shape)
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        logging.debug(data.shape)
        return data
