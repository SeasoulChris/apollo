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
            nn.Linear(feature_size, 30),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.mlp(X)

class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.BCELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        return
