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

import cv2 as cv
import math
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

sys.path.append('../../utilities')

from helper_utils import *
from IO_utils import *
from network_utils import *


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
                        print (feature_len-135)
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
        return (self.obs_features[idx], \
                self.lane_features[idx], \
                self.lane_features[idx].shape[0], \
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
    obs_features = obs_features[:,:45]
    lane_features_padded = lane_features_padded[idx_new]
    num_lanes = num_lanes[idx_new]
    traj_labels = traj_labels[idx_new]

    # Convert to torch tensors.
    return (torch.from_numpy(obs_features).view(N, 45), \
            torch.from_numpy(lane_features_padded).view(N, max_num_laneseq, 400), \
            torch.from_numpy(num_lanes).float()),\
           torch.from_numpy(traj_labels).view(N, 20)


#########################################################
# Network
#########################################################
class lane_scanning_model(torch.nn.Module):
    def __init__(self,\
                 dim_cnn=[4, 10, 16, 25],\
                 hidden_size = 34,\
                 dim_lane_fc = [34*8, 110, 56],\
                 dim_obs_fc = [45, 35, 23],\
                 dim_traj_fc = [79, 46, 20]):
        super(lane_scanning_model, self).__init__()

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
        self.single_lane_maxpool = nn.MaxPool1d(4)
        self.single_lane_dropout = nn.Dropout(0.0)

        self.input_embedding = generate_mlp(\
            [80, 256], dropout=0.0)

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        # TODO(jiacheng): add attention mechanisms to focus on some lanes.
        self.multi_lane_rnn = nn.LSTM(
            input_size=dim_cnn[3] * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True)

        self.multi_lane_fc = generate_mlp(\
            dim_lane_fc, dropout=0.0)

        self.obs_feature_fc = generate_mlp(\
            dim_obs_fc, dropout=0.0)

        self.traj_fc = generate_mlp(dim_traj_fc, \
            last_layer_nonlinear=False, dropout=0.0)

    def forward(self, X):
        obs_fea, lane_fea, num_laneseq = X
        N = obs_fea.size(0)
        max_num_laneseq = lane_fea.size(1)

        # Process each lane-sequence with CNN-1d
        # N x max_num_laneseq x 400
        lane_fea = lane_fea.view(N * max_num_laneseq, 100, 4)
        lane_fea = lane_fea.transpose(1, 2).float()
        # (N * max_num_laneseq) x 4 x 100
        lane_fea = self.single_lane_cnn(lane_fea)
        # (N * max_num_laneseq) x 25 x 8
        lane_fea = self.single_lane_maxpool(lane_fea)
        # (N * max_num_laneseq) x 25 x 2
        lane_fea = lane_fea.view(lane_fea.size(0), -1)
        lane_fea = self.single_lane_dropout(lane_fea)
        embed_dim = lane_fea.size(1)
        # (N * max_num_laneseq) x 50
        lane_fea = lane_fea.view(N, max_num_laneseq, embed_dim)

        # Process each obstacle's lane-sequences with RNN
        static_fea = pack_padded_sequence(lane_fea, num_laneseq, batch_first=True)
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        static_fea_states, _ = self.multi_lane_rnn(static_fea, (h0, c0))
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
        obs_fea_final = self.obs_feature_fc(obs_fea.float())
        fea_all = torch.cat((obs_fea_final, static_fea_final), 1)
        # N x 79

        # TODO(jiacheng): use RNN decoder to output traj
        # TODO(jiacheng): multi-modal prediction
        traj = self.traj_fc(fea_all)
        # N x 30

        return traj


#########################################################
# Loss
#########################################################
class lane_scanning_loss:
    def loss_fn(self, y_pred, y_true):
        #TODO(jiacheng): elaborate on this (e.g. consider final displacement error, etc.)
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true.float())

    def loss_info(self, y_pred, y_true):
        return


#########################################################
# Analyzer
#########################################################
class lane_scanning_analyzer:
    def __init__(self, top=40.0, bottom=10.0, left=20.0, right=20.0,\
                 resolution=0.1):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.resolution = resolution

        self.W = int((left+right)/resolution)
        self.H = int((top+bottom)/resolution)

        self.base_img = np.zeros([self.W, self.H, 3], dtype=np.uint8)

    def process(self, X, y, pred):
        # Convert data from cuda-torch to numpy format
        obs_features, lane_features, num_lane_seq = X
        # N x 45
        obs_features = obs_features.cpu().numpy()
        # N x max_num_lane_seq x 400
        lane_features = lane_features.cpu().numpy()
        # N x 1
        num_lane_seq = num_lane_seq.cpu().numpy()
        y = y.cpu().numpy()
        pred = pred.cpu().numpy()

        N = obs_features.shape[0]
        for idx in range(N):
            img = self.base_img
            # Based on X, plot the lane-graph ahead
            for lane_idx in range(num_lane_seq[idx]):
                curr_lane = lane_features[idx, lane_idx, :]

                for point_idx in range(99):
                    cv.line(img, point_to_idx(curr_lane[point_idx*4], curr_lane[point_idx*4+1]), \
                        point_to_idx(curr_lane[point_idx*4+4], curr_lane[point_idx*4+5]), \
                        color=[0, 0, 128], thickness=4)
            # Based on X, plot obstacle historic trajectory
            # TODO(jiacheng)

            # Based on y, plot the ground truth trajectory
            # TODO(jiacheng)

            # Based on pred, plot the predicted trajectory
            # TODO(jiacheng)
            cv.imwrite('img{}'.format(N), img)

        return

    def point_to_idx(self, point_x, point_y):
        return (int((point_x + self.left)/self.resolution),\
                int((point_y + self.bottom)/self.resolution))
