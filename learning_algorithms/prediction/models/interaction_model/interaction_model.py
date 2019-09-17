#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


dim_input = 3
dim_output = 1


class InteractionModel(nn.Module):
    def __init__(self, feature_size, delta):
        super(InteractionModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_size, 1, bias=False)
        )
        self._delta = delta

    def forward(self, x):
        out = self.mlp(x)
        out = torch.add(out, self._delta)
        out = F.relu(out)
        return out


class InteractionLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.L1Loss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        # TODO(kechxu) fix the following error
        # y_pred = y_pred.cpu()
        # loss = y_pred.type(torch.float).mean().item()
        # print("Loss is {:.3f} %".format(loss))
        print("----------- one epoch done -----------")
