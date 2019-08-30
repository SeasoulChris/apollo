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
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

from learning_algorithms.utilities.helper_utils import *
from learning_algorithms.utilities.IO_utils import *
from learning_algorithms.utilities.loss_utils import *
from learning_algorithms.utilities.network_utils import *


#########################################################
# Dataset
#########################################################
class LaneScanningDataset(Dataset):
    def __init__(self, dir, verbose=True):
        self.all_files = GetListOfFiles(dir)
        self.obs_features = []
        self.lane_features = []
        self.traj_labels = []
        # Each file contains an array of lists with each list having different sizes.
        # Go through each file:
        for i, file in enumerate(self.all_files):
            # Convert file content to list of lists
            file_content = np.load(file)
            file_content = file_content.tolist()
            # Go through each entry and, if data point is valid,
            # append to the lists above.
            for data_point in file_content:
                # Sanity checks.
                feature_len = data_point[0]
                if feature_len == 135:
                    if verbose:
                        print ('Skipped this one because it has no lane.')
                    continue
                if feature_len <= 0:
                    if verbose:
                        print ('Skipped this one because it has no feature.')
                    continue
                if (feature_len-135) % 400 != 0:
                    if verbose:
                        print ('Skipped this one because dimension isn\'t correct.')
                    continue

                curr_num_lanes = int((feature_len-135)/400)
                self.obs_features.append(
                    np.asarray(data_point[1:46]).reshape((1, 45)))
                self.lane_features.append(
                    np.asarray(data_point[136:-30]).reshape((curr_num_lanes, 400)))
                traj_label = np.zeros((1, 20))
                for j, point in enumerate(data_point[-30:-20]):
                    traj_label[0, j], traj_label[0, j+10] = \
                        world_coord_to_relative_coord(point, data_point[-30])
                self.traj_labels.append(traj_label)
            if verbose:
                print ('Loaded {} out of {} files'.format(i+1, len(self.all_files)))

        self.length = len(self.obs_features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.obs_features[idx],
                self.lane_features[idx],
                self.lane_features[idx].shape[0],
                self.traj_labels[idx])


def collate_with_padding(batch):
    # batch is a list of tuples, so unzip to form lists of np-arrays.
    obs_features, lane_features, num_lanes, traj_labels = zip(*batch)

    # Stack into numpy arrays.
    N = len(obs_features)
    max_num_laneseq = max(num_lanes)
    # N x 45
    obs_features = np.concatenate(obs_features, axis=0)
    # N x max_num_laneseq x 400
    lane_features_padded = np.zeros((N, max_num_laneseq, 400))
    for i, lane_fea in enumerate(lane_features):
        lane_features_padded[i, :num_lanes[i], :] = lane_fea
    # N x 1
    num_lanes = np.asarray(num_lanes).reshape((-1))
    # N x 30
    traj_labels = np.concatenate(traj_labels, axis=0)

    # Sort in descending order of available number of lanes
    idx_new = np.argsort(-num_lanes).tolist()
    obs_features = obs_features[idx_new]
    lane_features_padded = lane_features_padded[idx_new]
    num_lanes = num_lanes[idx_new]
    traj_labels = traj_labels[idx_new]

    # Convert to torch tensors.
    return (torch.from_numpy(obs_features).view(N, 45),
            torch.from_numpy(lane_features_padded).view(N, max_num_laneseq, 400),
            torch.from_numpy(num_lanes).float()),\
        torch.from_numpy(traj_labels).view(N, 20)


#########################################################
# Network
#########################################################
class lane_scanning_model(torch.nn.Module):
    def __init__(self,
                 dim_cnn=[4, 10, 16, 25],
                 hidden_size=128,
                 dim_lane_fc=[128*8, 700, 456, 230],
                 dim_obs_fc=[45, 38, 32],
                 dim_traj_fc=[262, 120, 40]):
        super(lane_scanning_model, self).__init__()
        self.dim_obs_fc = dim_obs_fc

        self.single_lane_cnn = torch.nn.Sequential(
            # L=100
            nn.Conv1d(dim_cnn[0], dim_cnn[1], 4, stride=2),
            nn.ReLU(),
            # L=49
            nn.Conv1d(dim_cnn[1], dim_cnn[2], 3, stride=2),
            nn.ReLU(),
            # L=24
            nn.Conv1d(dim_cnn[2], dim_cnn[3], 3, stride=3),
            nn.ReLU(),
            # L=8
        )
        self.single_lane_maxpool = nn.MaxPool1d(1)
        self.single_lane_dropout = nn.Dropout(0.0)

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        # TODO(jiacheng): add attention mechanisms to focus on some lanes.
        self.multi_lane_rnn = nn.LSTM(
            input_size=dim_cnn[-1] * 8 + dim_obs_fc[-1],
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True)

        self.multi_lane_fc = generate_mlp(
            dim_lane_fc, dropout=0.0)

        self.obs_feature_fc = generate_mlp(
            dim_obs_fc, dropout=0.0)

        self.traj_fc = generate_mlp(dim_traj_fc,
                                    last_layer_nonlinear=False, dropout=0.0)

    def forward(self, X):
        obs_fea, lane_fea, num_laneseq = X
        N = obs_fea.size(0)
        max_num_laneseq = lane_fea.size(1)

        # Process each lane-sequence with CNN-1d
        # N x max_num_laneseq x 400
        lane_fea = lane_fea.view(N * max_num_laneseq, 100, 4)
        lane_fea = lane_fea.transpose(1, 2).float()
        lane_fea_original = lane_fea
        # (N * max_num_laneseq) x 4 x 100
        lane_fea = self.single_lane_cnn(lane_fea)
        # (N * max_num_laneseq) x 25 x 8
        lane_fea = self.single_lane_maxpool(lane_fea)
        # (N * max_num_laneseq) x 25 x 8
        lane_fea = lane_fea.view(lane_fea.size(0), -1)
        lane_fea = self.single_lane_dropout(lane_fea)
        embed_dim = lane_fea.size(1)
        # (N * max_num_laneseq) x 200
        lane_fea = lane_fea.view(N, max_num_laneseq, embed_dim)
        # N x max_num_laneseq x 200
        lane_fea = lane_fea + lane_fea_original[:, :2, :].view(N, max_num_laneseq, 200)

        obs_fea_original = obs_fea
        obs_fea = self.obs_feature_fc(obs_fea.float())
        # N x 32
        obs_fea = obs_fea.view(N, 1, self.dim_obs_fc[-1]).repeat(1, max_num_laneseq, 1)
        # N x max_num_laneseq x 32

        static_fea = torch.cat((lane_fea, obs_fea), 2)

        # Process each obstacle's lane-sequences with RNN
        static_fea = pack_padded_sequence(static_fea, num_laneseq, batch_first=True)
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        static_fea_states, _ = self.multi_lane_rnn(static_fea, (h0, c0))
        static_fea, _ = pad_packed_sequence(static_fea_states, batch_first=True)
        # N x max_num_laneseq x 256
        min_val = torch.min(static_fea)
        static_fea_maxpool, _ = pad_packed_sequence(
            static_fea_states, batch_first=True, padding_value=min_val.item()-0.1)
        static_fea_maxpool, _ = torch.max(static_fea_maxpool, 1)
        static_fea_avgpool = torch.sum(static_fea, 1) / num_laneseq.reshape(N, 1)
        static_fea_front = static_fea[:, 0]
        idx_1 = np.arange(N).tolist()
        idx_2 = (num_laneseq-1).int().data.tolist()
        static_fea_back = static_fea[idx_1, idx_2]
        static_fea_all = torch.cat((static_fea_maxpool, static_fea_avgpool,
                                    static_fea_front, static_fea_back), 1)
        # N x (256 * 4) = N x 1024

        static_fea_final = self.multi_lane_fc(static_fea_all)
        obs_fea_final = self.obs_feature_fc(obs_fea_original.float())
        fea_all = torch.cat((obs_fea_final, static_fea_final), 1)
        # N x 79

        # TODO(jiacheng): use RNN decoder to output traj
        # TODO(jiacheng): multi-modal prediction
        traj = self.traj_fc(fea_all)
        traj = traj.view(N, 2, 20)
        # N x 2 x 20

        return traj


#########################################################
# Loss
#########################################################
class lane_scanning_loss:
    def loss_fn(self, y_pred, y_true):
        # TODO(jiacheng): elaborate on this (e.g. consider final displacement error, etc.)
        y_true = y_true.float()
        return multi_modal_loss(y_pred, y_true)

    def loss_helper(self, y_pred, y_true):
        loss_func = nn.MSELoss(reduction='none')
        loss_result = loss_func(y_pred, y_true)
        loss_result = torch.mean(loss_result, dim=1)

    def loss_info(self, y_pred, y_true):
        return None
