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

import argparse
import copy
import logging
import numpy as np
import os
import torch

from learning_algorithms.prediction.datasets.apollo_vehicle_regroad_dataset.apollo_vehicle_regroad_dataset \
    import *
from learning_algorithms.prediction.models.lane_attention_model.lane_attention_model import *
from learning_algorithms.utilities.network_utils import *
from learning_algorithms.utilities.train_utils import *


class OnlinePredictionLayer(nn.Module):
    def __init__(self):
        super(OnlinePredictionLayer, self).__init__()
        self.mlp = None

    def forward(self, X):
        return self.mlp(X)


if __name__ == "__main__":
    offline_model = CruiseMLP()
    offline_model.load_state_dict(torch.load(
        '/home/jiacheng/work/apollo-prophet/learning_algorithms/pipelines/vehicle_regroad_classification_pipeline/model_epoch11_valloss0.628565.pt'))

    online_obs_enc = offline_model.vehicle_encoding
    online_obs_enc.eval()
    traced_online_obs_enc = torch.jit.trace(online_obs_enc, torch.zeros([1, 180]))
    traced_online_obs_enc.save('./traced_online_obs_enc.pt')

    online_lane_enc = offline_model.lane_encoding
    online_lane_enc.eval()
    traced_online_lane_enc = torch.jit.trace(online_lane_enc, torch.zeros([1, 400]))
    traced_online_lane_enc.save('./traced_online_lane_enc.pt')

    online_pred_layer = OnlinePredictionLayer()
    online_pred_layer.mlp = copy.deepcopy(offline_model.prediction_layer.mlp)
    online_pred_layer.eval()
    traced_online_pred_layer = torch.jit.trace(online_pred_layer, torch.zeros([1, 257]))
    traced_online_pred_layer.save('./traced_online_pred_layer.pt')
