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

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def world_coord_to_relative_coord(input_world_coord, ref_world_coord):
    x_diff = input_world_coord[0] - ref_world_coord[0]
    y_diff = input_world_coord[1] - ref_world_coord[1]
    rho = math.sqrt(x_diff ** 2 + y_diff ** 2)
    theta = math.atan2(y_diff, x_diff) - ref_world_coord[2]

    return math.cos(theta)*rho, math.sin(theta)*rho

'''
@brief Apply zero-padding so that the input tensor to the model is of uniform size.
@input features: a list of features (might be of different sizes)
@output: a numpy matrix with features of equal sizes (padded with zero)
@output: a list of the actual number of lane-sequences for each entry.
'''
def preprocess_features(features):
    features = features.tolist()
    max_feature_len = 4445
    feature_lengths = [feature[0] for feature in features]
    num_laneseq = [int((feature[0]-45)/400) for feature in features]

    N = len(features)
    padded_features = np.zeros((N, max_feature_len+1))
    labels = np.zeros((N, 20))
    num_skipped = 0

    for i, feature_len in enumerate(feature_lengths):
        if feature_len > max_feature_len:
            # print ('Skipped this one with {} lane seqs.'\
            #        .format(int((feature_len-45)/400)))
            num_skipped += 1
            continue
        if feature_len == 45:
            # print ('Skipped this one because it has no lane.')
            num_skipped += 1
            continue
        if feature_len <= 0:
            # print ('Skipped this one because it has no feature.')
            num_skipped += 1
            continue
        if (feature_len-45) % 400 != 0:
            # print ('Skipped this one because dimension isn\'t correct.')
            num_skipped += 1
            continue
        padded_features[i-num_skipped, 0:feature_len] = \
            features[i][1:feature_len+1]
        padded_features[i-num_skipped, -1] = num_laneseq[i]
        for j, point in enumerate(features[i][-30:-20]):
            labels[i-num_skipped, j], labels[i-num_skipped, j+10] =\
                world_coord_to_relative_coord(point, features[i][-30])
        # sanity check:
        if labels[i-num_skipped, 0] != 0.0 or labels[i-num_skipped, 10] != 0.0:
            print ('Future trajectory isn\'t correctly normalized.')

    if num_skipped == 0:
        return padded_features, labels
    else:
        return padded_features[:-num_skipped, :], labels[:-num_skipped, :]

def batch_preprocess(X, y):
    idx_1 = np.arange(X.shape[0]).tolist()
    idx_2 = torch.argsort(X[:,-1], descending=True).tolist()
    return X[idx_2], y[idx_2]

class lane_scanning_model(torch.nn.Module):
    def __init__(self):
        super(lane_scanning_model, self).__init__()

        self.single_lane_cnn = torch.nn.Sequential(
            # L=4x100
            nn.Conv1d(4, 10, 4, stride=2),
            nn.ReLU(),
            # L=10x49
            nn.Conv1d(10, 16, 3, stride=2),
            nn.ReLU(),
            # L=16x24
            nn.Conv1d(16, 25, 3, stride=3),
            nn.ReLU(),
            # L=25x8
        )
        self.single_lane_maxpool = nn.MaxPool1d(4)
        self.single_lane_dropout = nn.Dropout(0.0)

        self.multi_lane_rnn = nn.GRU(
            input_size=50,
            hidden_size=34,
            bidirectional=True,
            batch_first=True)
        self.multi_lane_fc = torch.nn.Sequential(
            nn.Linear(68 * 4, 110),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(110, 56),
            nn.ReLU(),
            nn.Dropout(0.0),
        )

        self.obs_feature_fc = torch.nn.Sequential(
            nn.Linear(45, 35),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(35, 23),
            nn.ReLU(),
            nn.Dropout(0.0),
        )

        self.traj_fc = torch.nn.Sequential(
            nn.Linear(79, 46),
            nn.ReLU(),
            nn.Linear(46, 20)
        )

    def forward(self, X):
        inputs = X[:,:-1]           # N x (45 + 400 * max_num_laneseq)
        num_laneseq = X[:,-1]       # N

        obs_in = inputs[:, :45]         # N x 45
        lane_in = inputs[:, 45:]        # N x (400 * max_num_laneseq)
        N = lane_in.size(0)
        max_num_laneseq = int(lane_in.size(1) / 400)

        # N x (max_num_laneseq * 100 * 4)
        lane_in = lane_in.contiguous()
        lane_in = lane_in.view(N * max_num_laneseq, 100, 4)
        lane_in = lane_in.transpose(1, 2)
        # (N * max_num_laneseq) x 4 x 100
        lane_fea = self.single_lane_cnn(lane_in)
        # (N * max_num_laneseq) x 25 x 8
        lane_fea = self.single_lane_maxpool(lane_fea)
        # (N * max_num_laneseq) x 25 x 2
        lane_fea = lane_fea.view(N*max_num_laneseq, -1)
        lane_fea = self.single_lane_dropout(lane_fea)
        lane_fea = lane_fea.view(N, max_num_laneseq, -1)
        # N x max_num_laneseq x 50

        # TODO(jiacheng): also train the initial hidden state
        static_fea = pack_padded_sequence(lane_fea, num_laneseq, batch_first=True)
        static_fea_states, _ = self.multi_lane_rnn(static_fea)
        static_fea, _ = pad_packed_sequence(static_fea_states, batch_first=True)
        # N x max_num_laneseq x 68

        min_val = torch.min(static_fea)
        static_fea_maxpool, _ = pad_packed_sequence(
            static_fea_states, batch_first=True, padding_value=min_val.item()-0.1)
        static_fea_maxpool, _ = torch.max(static_fea_maxpool, 1)
        static_fea_avgpool = torch.sum(static_fea, 1) / num_laneseq.reshape(N,1)
        static_fea_front = static_fea[:,0]
        idx_1 = np.arange(N).tolist()
        idx_2 = (num_laneseq-1).int().data.tolist()
        static_fea_back = static_fea[idx_1, idx_2]
        static_fea_all = torch.cat((static_fea_maxpool, static_fea_avgpool, \
            static_fea_front, static_fea_back), 1)
        # N x (68 * 4) = N x 272

        static_fea_final = self.multi_lane_fc(static_fea_all)
        obs_fea_final = self.obs_feature_fc(obs_in)
        fea_all = torch.cat((obs_fea_final, static_fea_final), 1)
        # N x 79

        # TODO(jiacheng): use RNN decoder to output traj
        # TODO(jiacheng): multi-modal prediction
        traj = self.traj_fc(fea_all)
        # N x 20

        return traj


class lane_scanning_loss:
    def loss_fn(self, y_pred, y_true):
        #TODO(jiacheng): elaborate on this
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        return
