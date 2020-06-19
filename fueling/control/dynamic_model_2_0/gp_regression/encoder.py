#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import fueling.common.logging as logging


class Encoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        super().__init__()
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, stride=4)
        self.fc = nn.Linear(250, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # original data shape: [sequency/window_size, batch_size, channel]
        # conv_input shape: [batch_size, channel, sequency/window_size]
        conv1_input = torch.transpose(torch.transpose(data, -2, -3), -2, -1)
        data = F.relu(self.conv1(conv1_input))
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        return data


class DummyEncoder(nn.Module):
    """encoder (for place holder only)"""

    def __init__(self):
        """Network initialization"""
        super().__init__()

    def forward(self, data):
        """Define forward computation and activation functions"""
        return data


class DilatedEncoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        super().__init__()
        # (5, 200)
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, dilation=5, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, dilation=1, stride=4)
        # set 200 to 250 for dilation = 3 or 2
        self.fc = nn.Linear(200, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # original data shape: [sequency/window_size, batch_size, channel]
        # conv_input shape: [batch_size, channel, sequency/window_size]
        conv1_input = torch.transpose(torch.transpose(data, -2, -3), -2, -1)
        data = F.relu(self.conv1(conv1_input))
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        return data
