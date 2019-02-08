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


dim_input = 7 + 72
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
        out = torch.mul(out, X[:,7:79:6])
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
