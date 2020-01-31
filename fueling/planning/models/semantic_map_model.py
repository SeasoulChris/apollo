#!/usr/bin/env python

import glob
import os

import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
# from torchvision import transforms

from fueling.common.coord_utils import CoordUtils

'''
========================================================================
Model definition
========================================================================
'''


class SemanticMapModel(nn.Module):
    def __init__(self, num_pred_points, num_history_points,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(SemanticMapModel, self).__init__()

        self.cnn = cnn_net(pretrained=pretrained)
        self.num_pred_points = num_pred_points
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features + num_history_points * 2, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, num_pred_points * 2)
        )

    def forward(self, X):
        img, obs_pos, _, _ = X
        out = self.cnn(img)
        out = out.view(out.size(0), -1)
        obs_pos = obs_pos.view(obs_pos.size(0), -1)
        out = torch.cat([out, obs_pos], -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.num_pred_points, 2)
        return out


class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        out = y_pred - y_true
        out = torch.sqrt(torch.sum(out ** 2, 2))
        out = torch.mean(out)
        return out
