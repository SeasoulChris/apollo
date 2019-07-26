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

import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *


class SelfLSTM(nn.Module):
    def __init__(self, pred_len=29, embed_size=64, hidden_size=128):
        super(SelfLSTM, self).__init__()
        self.pred_len = pred_len
        self.disp_embed_size = embed_size
        self.hidden_size = hidden_size

        self.disp_embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size, 5),
        )

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_pos: N x 20 x 2
            - obs_pos_rel: N x 20 x 2
            - lane_features: M x 150 x 4
            - same_obs_mask: M x 1
        '''
        obs_hist_size, obs_pos, obs_pos_rel, lane_features, same_obs_mask = X
        N = obs_pos.size(0)
        observation_len = obs_pos.size(1)
        ht, ct = self.h0.repeat(N, 1), self.h0.repeat(N, 1)

        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))
        ts_obs_mask, curr_obs_pos, curr_obs_pos_rel = None, None, None
        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                ts_obs_mask = (obs_hist_size > observation_len-t).long().view(-1)
                curr_obs_pos_rel = obs_pos_rel[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                ts_obs_mask = pred_mask
                pred_out[:, t-observation_len, :] = self.pred_layer(ht.clone()).float().clone()
                curr_obs_pos_rel = pred_out[:, t-observation_len, :2]
                curr_obs_pos = curr_obs_pos + curr_obs_pos_rel
                pred_traj[:, t-observation_len, :] = curr_obs_pos.clone()

            curr_N = torch.sum(ts_obs_mask).long().item()
            if curr_N == 0:
                continue

            ts_obs_mask = (ts_obs_mask == 1)
            disp_embedding = self.disp_embed((curr_obs_pos_rel[ts_obs_mask,:]).clone()).view(curr_N, 1, -1)
            _, (ht_new, ct_new) = self.lstm(
            	disp_embedding, (ht[ts_obs_mask, :].view(1,curr_N,-1), ct[ts_obs_mask, :].view(1,curr_N,-1)))
            ht[ts_obs_mask, :] = ht_new.view(curr_N, -1)
            ct[ts_obs_mask, :] = ct_new.view(curr_N, -1)

        return pred_out, pred_traj
