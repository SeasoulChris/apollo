#!/usr/bin/env python

from collections import OrderedDict

import torch
import torch.nn as nn


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.raw_feature_extraction = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, 64, 64, padding=32)),
            ('conv2', nn.Conv1d(64, 64, 64, padding=31)),
            ('pool1', nn.MaxPool1d(220))
        ]))
        self.classification = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(1, 32, kernel_size=4, padding=2)),
            ('conv4', nn.Conv2d(32, 32, kernel_size=4, padding=1)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv5', nn.Conv2d(32, 64, kernel_size=4, padding=2)),
            ('conv6', nn.Conv2d(64, 64, kernel_size=4, padding=1)),
            ('pool3', nn.MaxPool2d(2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(37888, 512)),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 64)),
            ('dropout2', nn.Dropout(0.5)),
            ('output', nn.Linear(64, 3)),
            ('softmax', nn.Softmax()),
        ]))

    def forward(self, X):
        raw_features = self.raw_feature_extraction(X)
        raw_features.unsqueeze_(1)
        out = self.classification(raw_features)
        return out


class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=4, padding=2)),
            ('bn1', nn.BatchNorm2d(32)),
            ('conv2', nn.Conv2d(32, 32, kernel_size=4, padding=1)),
            ('bn2', nn.BatchNorm2d(32)),
            ('pool1', nn.MaxPool2d(2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(53248, 512)),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 64)),
            ('dropout2', nn.Dropout(0.5)),
            ('output', nn.Linear(64, 3)),
            ('softmax', nn.Softmax()),
        ]))

    def forward(self, X):
        X.unsqueeze_(1)
        out = self.cnn(X)
        return out


if __name__ == '__main__':
    input = torch.ones([1, 104, 65])
    model = MLNet()
    print(model(input))
