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


class CruiseMLP(nn.Module):
    def __init__(self):
        super(CruiseMLP, self).__init__()
        self.vehicle_encoding = VehicleLSTM(embed_size=64, hidden_size=128, encode_size=128)
        self.lane_encoding = LaneLSTM(embed_size=64, hidden_size=128, encode_size=127)
        self.prediction_layer = DistributionalScoring(obs_enc_size=128, lane_enc_size=128, aggr_enc_size=1, mlp_size=[100, 16, 1])

    def forward(self, X):
        obs_features, obs_hist_size, lane_features, same_obs_mask = X
        obs_features = obs_features.float()
        obs_hist_size = obs_hist_size.float()
        lane_features = lane_features.float()
        same_obs_mask = same_obs_mask.float()
        N = obs_features.size(0)
        M = lane_features.size(0)

        obs_enc = self.vehicle_encoding(obs_features, obs_hist_size)

        lane_enc_1 = self.lane_encoding(lane_features)
        lane_enc_2 = lane_features[:, 0].view(M, 1)
        lane_enc = torch.cat((lane_enc_1, lane_enc_2), 1)

        aggr_enc = cuda(torch.zeros(N, 1))

        out = self.prediction_layer(obs_enc, lane_enc, aggr_enc, same_obs_mask)

        return out


class LaneAttention(nn.Module):
    def __init__(self):
        super(LaneAttention, self).__init__()
        self.vehicle_encoding = VehicleLSTM(embed_size=64, hidden_size=128, encode_size=128)
        self.backward_lane_encoding = None
        self.lane_encoding = LaneLSTM(embed_size=64, hidden_size=128, encode_size=127)
        self.lane_aggregation = AttentionalAggregation(input_encoding_size=128, output_size=512)
        self.prediction_layer = DistributionalScoring(obs_enc_size=128, lane_enc_size=128, aggr_enc_size=1024, mlp_size=[100, 30, 1])

    def forward(self, X):
        obs_features, obs_hist_size, lane_features, same_obs_mask = X
        obs_features = obs_features.float()
        obs_hist_size = obs_hist_size.float()
        lane_features = lane_features.float()
        same_obs_mask = same_obs_mask.float()
        N = obs_features.size(0)
        M = lane_features.size(0)

        obs_enc = self.vehicle_encoding(obs_features, obs_hist_size)

        lane_enc_1 = self.lane_encoding(lane_features)
        lane_enc_2 = lane_features[:, 0].view(M, 1)
        lane_enc = torch.cat((lane_enc_1, lane_enc_2), 1)

        aggr_enc = self.lane_aggregation(obs_enc, lane_enc, same_obs_mask)

        out = self.prediction_layer(obs_enc, lane_enc, aggr_enc, same_obs_mask)

        return out


# TODO(jiacheng):
#   - Only run through the actual time-stamps with the help of "packing".
class VehicleDynamicLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(VehicleDynamicLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(8, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.vehicle_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size*8, encode_size),
            nn.ReLU(),
        )

    def forward(self, obs_features, hist_size):
        '''Forward function
            - obs_features: N x 180
            - hist_size: N x 1

            output: N x encode_size
        '''
        N = obs_features.size(0)

        # Input embedding.
        # (N x 20 x 1)
        obs_x = obs_features[:, 1::9].view(N, 20, 1)
        obs_y = obs_features[:, 2::9].view(N, 20, 1)
        vel_x = obs_features[:, 3::9].view(N, 20, 1)
        vel_y = obs_features[:, 4::9].view(N, 20, 1)
        acc_x = obs_features[:, 5::9].view(N, 20, 1)
        acc_y = obs_features[:, 6::9].view(N, 20, 1)
        vel_heading = obs_features[:, 7::9].view(N, 20, 1)
        vel_heading_changing_rate = obs_features[:, 8::9].view(N, 20, 1)
        # (N x 20 x 8)
        obs_position = torch.cat((obs_x, obs_y, vel_x, vel_y, acc_x, acc_y, \
            vel_heading, vel_heading_changing_rate), 2).float()
        # (N x 20 x embed_size)
        obs_embed = self.embed(obs_position.view(N*20, 8)).view(N, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        # (N x 20 x 2*hidden_size)
        obs_states, _ = self.vehicle_rnn(obs_embed, (h0, c0))
        # (N x 2*hidden_size)
        front_states = obs_states[:, 0, :]
        back_states = obs_states[:, -1, :]
        max_states, _ = torch.max(obs_states, 1)
        avg_states = torch.mean(obs_states, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, avg_states), 1)
        out = self.encode(out)
        return out


class VehicleLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(VehicleLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(8, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.vehicle_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size*8, encode_size),
            nn.ReLU(),
        )

    def forward(self, obs_features, hist_size):
        '''Forward function
            - obs_features: N x 180
            - hist_size: N x 1

            output: N x encode_size
        '''
        N = obs_features.size(0)

        # Input embedding.
        # (N x 20 x 1)
        obs_x = obs_features[:, 1::9].view(N, 20, 1)
        obs_y = obs_features[:, 2::9].view(N, 20, 1)
        vel_x = obs_features[:, 3::9].view(N, 20, 1)
        vel_y = obs_features[:, 4::9].view(N, 20, 1)
        acc_x = obs_features[:, 5::9].view(N, 20, 1)
        acc_y = obs_features[:, 6::9].view(N, 20, 1)
        vel_heading = obs_features[:, 7::9].view(N, 20, 1)
        vel_heading_changing_rate = obs_features[:, 8::9].view(N, 20, 1)
        # (N x 20 x 8)
        obs_position = torch.cat((obs_x, obs_y, vel_x, vel_y, acc_x, acc_y, \
            vel_heading, vel_heading_changing_rate), 2).float()
        # (N x 20 x embed_size)
        obs_embed = self.embed(obs_position.view(N*20, 8)).view(N, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        # (N x 20 x 2*hidden_size)
        obs_states, _ = self.vehicle_rnn(obs_embed, (h0, c0))
        # (N x 2*hidden_size)
        front_states = obs_states[:, 0, :]
        back_states = obs_states[:, -1, :]
        max_states, _ = torch.max(obs_states, 1)
        avg_states = torch.mean(obs_states, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, avg_states), 1)
        out = self.encode(out)
        return out


# TODO(jiacheng):
#   - Implement this.
class BackwardLaneLSTM(nn.Module):
    def __init__(self):
        super(BackwardLaneLSTM, self).__init__()

    def forward(self, X):
        return X


# TODO(jiacheng):
#   - With the help of encoded obstacle_feature, add attention mechanism in LaneLSTM.
class AttentionalLaneLSTM(nn.Module):
    def __init__(self):
        super(AttentionalLaneLSTM, self).__init__()

    def forward(self, X):
        return X


class LaneLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(LaneLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.single_lane_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size*8, encode_size),
            nn.ReLU(),
        )

    def forward(self, lane_features):
        '''Forward function:
            - lane_features: N x 400

            output: N x encode_size
        '''
        N = lane_features.size(0)

        # Input embedding.
        # (N x 100 x embed_size)
        lane_embed = self.embed(lane_features.float().view(N*100, 4)).view(N, 100, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        # (N x 20 x 2*hidden_size)
        lane_states, _ = self.single_lane_rnn(lane_embed, (h0, c0))
        # (N x 2*hidden_size)
        front_states = lane_states[:, 0, :]
        back_states = lane_states[:, -1, :]
        max_states, _ = torch.max(lane_states, 1)
        avg_states = torch.mean(lane_states, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, avg_states), 1)
        out = self.encode(out)
        return out


# TODO(jiacheng):
#   - Add pairwise attention between obs_encoding and every lane_encoding during aggregating.
class AttentionalAggregation(nn.Module):
    def __init__(self, input_encoding_size=128, output_size=512):
        super(AttentionalAggregation, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.output_size = output_size

        self.encode = torch.nn.Sequential(
            nn.Linear(input_encoding_size, output_size),
            nn.ReLU(),
        )

    def forward(self, obs_encoding, lane_encoding, same_obs_mask):
        '''Forward function
            - obs_encoding: N x input_encoding_size
            - lane_encoding: M x input_encoding_size
            - same_obs_mask: M x 1

            output: N x output_size
        '''
        N = obs_encoding.size(0)
        out = cuda(torch.zeros(N, self.output_size*2))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()

            # (curr_num_lane x input_encoding_size)
            curr_lane_encoding = lane_encoding[curr_mask, :].view(curr_num_lane, -1)
            curr_lane_encoding = self.encode(curr_lane_encoding)
            curr_lane_maxpool, _ = torch.max(curr_lane_encoding, 0, keepdim=True)
            curr_lane_avgpool = torch.mean(curr_lane_encoding, 0, keepdim=True)
            out[obs_id, :] = torch.cat((curr_lane_maxpool, curr_lane_avgpool), 1)

        return out


class DistributionalScoring(nn.Module):
    def __init__(self, obs_enc_size=128, lane_enc_size=128, aggr_enc_size=1024, mlp_size=[600, 100, 16, 1]):
        super(DistributionalScoring, self).__init__()
        self.mlp = generate_mlp(\
            [obs_enc_size+lane_enc_size+aggr_enc_size] + mlp_size, dropout=0.0, last_layer_nonlinear=False)
        self.softmax_layer = nn.Softmax()

    def forward(self, obs_encoding, lane_encoding, aggregated_info, same_obs_mask):
        '''Forward function (M >= N)
            - obs_encoding: N x obs_enc_size
            - lane_encoding: M x lane_enc_size
            - aggregated_info: N x aggr_enc_size
        '''
        N = obs_encoding.size(0)
        M = lane_encoding.size(0)
        out = cuda(torch.zeros(M, 1))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()

            curr_obs_enc = obs_encoding[obs_id, :].view(1, -1)
            curr_obs_enc = curr_obs_enc.repeat(curr_num_lane, 1)
            curr_agg_info = aggregated_info[obs_id, :].view(1, -1)
            curr_agg_info = curr_agg_info.repeat(curr_num_lane, 1)
            curr_lane_enc = lane_encoding[curr_mask, :].view(curr_num_lane, -1)

            curr_encodings = torch.cat((curr_obs_enc, curr_agg_info, curr_lane_enc), 1)
            curr_scores = self.mlp(curr_encodings).view(-1)
            curr_scores = self.softmax_layer(curr_scores)
            out[curr_mask, :] = curr_scores.view(curr_num_lane, 1)

        return out


class ClassificationLoss:
    def loss_fn(self, y_pred, y_true_tuple):
        y_labels, y_is_cutin, y_same_obs_mask = y_true_tuple
        y_pred_score = -torch.log(y_pred).float()
        y_labels = y_labels.float()

        total_num_data_pt = y_same_obs_mask.max().long().item() + 1
        scores = cuda(torch.zeros(total_num_data_pt))

        for obs_id in range(total_num_data_pt):
            curr_mask = (y_same_obs_mask[:, 0] == obs_id)
            curr_pred = y_pred_score[curr_mask, 0]
            curr_true = y_labels[curr_mask, 0]
            curr_score = torch.sum(curr_pred * curr_true)
            scores[obs_id] = curr_score

        final_loss = torch.mean(scores)
        return final_loss

    def loss_info(self, y_pred, y_true_tuple):
        y_labels, y_is_cutin, y_same_obs_mask = y_true_tuple
        total_num_data_pt = 0.0
        total_correct_data_pt = 0.0

        for obs_id in range(y_same_obs_mask.max().long().item() + 1):
            curr_mask = (y_same_obs_mask[:, 0] == obs_id)

            curr_pred = y_pred[curr_mask, 0]
            curr_true = y_labels[curr_mask, 0]
            if curr_true[torch.argmax(curr_pred)] == 1:
                total_correct_data_pt += 1
            total_num_data_pt += 1

        return total_correct_data_pt/total_num_data_pt
