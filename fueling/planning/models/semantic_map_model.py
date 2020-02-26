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
        # fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            # nn.Linear(fc_in_features + num_history_points * 2, 500),
            # TODO(all): fix input size with real data
            nn.Linear(62734, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, num_pred_points * 2)
        )

    def forward(self, X):
        img, features = X
        # print("features size:{}".format(features.size()))
        # print("image size :{}".format(img.size()))
        out = self.cnn(img)
        # print("cnn out size :{}".format(out.size()))
        out = out.view(out.size(0), -1)
        # print("cnn out after view:{}".format(out.size()))

        features = features.view(features.size(0), -1)
        # print("features size after view:{}".format(features.size()))

        out = torch.cat([out, features], -1)
        # print("size of out: {}".format(out.size()))

        # features size:torch.Size([2, 14])
        # image size :torch.Size([2, 3, 224, 224])
        # cnn out size :torch.Size([2, 1280, 7, 7])
        # cnn out after view:torch.Size([2, 62720])
        # features size after view:torch.Size([2, 14])
        # size of out: torch.Size([2, 62734])

        out = self.fc(out)
        out = out.view(out.size(0), self.num_pred_points, 2)
        return out


class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        # print("y_pred size: {}".format(y_pred.size()))
        # print("y_true size: {}".format(y_true.size()))
        y_true = y_true.view(y_true.size(0), -1, 2)
        # print("y_true size after view: {}".format(y_true.size()))

        # y_pred size: torch.Size([2, 80, 2])
        # y_true size: torch.Size([2, 160])
        # y_true size after view: torch.Size([2, 80, 2])

        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        # print("y_pred size: {}".format(y_pred.size()))
        # print("y_true size: {}".format(y_true.size()))
        y_true = y_true.view(y_true.size(0), -1, 2)
        # print("y_true size after view: {}".format(y_true.size()))

        out = y_pred - y_true
        out = torch.sqrt(torch.sum(out ** 2, 2))
        out = torch.mean(out)
        return out
