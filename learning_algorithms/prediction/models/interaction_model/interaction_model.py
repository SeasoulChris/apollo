###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import glob
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        return
