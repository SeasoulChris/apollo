#!/usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


dim_input = 114
dim_output = 12

'''
========================================================================
Model definition
========================================================================
'''


class JunctionMLPModel(nn.Module):
    def __init__(self, feature_size):
        super(JunctionMLPModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_size, 60),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(30, 12),
            # nn.Softmax()
        )

    def forward(self, X):
        out = self.mlp(X)
        out = torch.mul(out, X[:, 18:114:8])
        out = F.softmax(out)
        return out


class JunctionMLPLoss():
    def loss_fn(self, y_pred, y_true):
        true_label = y_true.topk(1)[1].view(-1)
        loss_func = nn.CrossEntropyLoss()
        return loss_func(y_pred, true_label)

    def loss_info(self, y_pred, y_true):
        y_pred = torch.from_numpy(np.concatenate(y_pred))
        y_true = y_true.cpu()
        pred_label = y_pred.topk(1)[1]
        true_label = y_true.topk(1)[1]
        accuracy = (pred_label == true_label).type(torch.float).mean().item()
        print("Accuracy is {:.3f} %".format(100*accuracy))
        return
