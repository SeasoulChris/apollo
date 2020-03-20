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


class TrajectoryImitationModel(nn.Module):
    def __init__(self,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(TrajectoryImitationModel, self).__init__()
        # compressed to 3 channel
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(62720, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, 30 * 5)
        )

    def forward(self, X):
        img = X
        out = self.compression_cnn_layer(img)
        # print("image size :{}".format(img.size()))
        out = self.cnn(out)
        # print("cnn out size :{}".format(out.size()))
        out = out.view(out.size(0), -1)
        # print("cnn out after view:{}".format(out.size()))

        # image size :torch.Size([2, , 501, 501])
        # cnn out size :torch.Size([2, 1280, 7, 7])
        # cnn out after view:torch.Size([2, 62720])

        out = self.fc(out)
        out = out.view(out.size(0), 30 * 5)
        return out


class TrajectoryImitationLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        # print("y_pred size: {}".format(y_pred.size()))
        # print("y_true size: {}".format(y_true.size()))
        y_true = y_true.view(y_true.size(0), -1)
        # print("y_true size after view: {}".format(y_true.size()))

        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        # print("y_pred size: {}".format(y_pred.size()))
        # print("y_true size: {}".format(y_true.size()))
        y_true = y_true.view(y_true.size(0), -1)
        # print("y_true size after view: {}".format(y_true.size()))

        out = y_pred - y_true
        out = torch.sqrt(torch.sum(out ** 2, 1))
        out = torch.mean(out)
        return out
