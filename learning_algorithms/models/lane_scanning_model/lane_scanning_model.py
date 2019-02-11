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
from torch.utils.data import Dataset

from utilities.helper_utils import *
from utilities.IO_utils import *


'''
========================================================================
Dataset set-up
========================================================================
'''
class LaneScanningDataset(Dataset):
    def __init__(self, dir, verbose=False):
        self.all_files = GetListOfFiles(dir)
        self.obs_features = []
        self.lane_features = []
        #self.lane_start_end = []
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
                feature_len = data_point[0]
                if feature_len == 45:
                    if verbose:
                        print ('Skipped this one because it has no lane.')
                    continue
                if feature_len <= 0:
                    if verbose:
                        print ('Skipped this one because it has no feature.')
                    continue
                if (feature_len-45) % 400 != 0:
                    if verbose:
                        print ('Skipped this one because dimension isn\'t correct.')
                    continue

                curr_num_lanes = int((feature_len-45)/400)
                self.obs_features.append(
                    np.asarray(data_point[1:46]).reshape((1, 45)))
                self.lane_features.append(
                    np.asarray(data_point[46:-30]).reshape((curr_num_lanes, 400)))
                #self.lane_start_end.append(
                #    (lane_start_idx, lane_start_idx+curr_num_lanes))
                traj_label = np.zeros((1, 30))
                for j, point in enumerate(data_point[-30:-15]):
                    traj_label[0, j], traj_label[0, j+15] = \
                        world_coord_to_relative_coord(point, data_point[-30])
                self.traj_labels.append(traj_label)
            print ('Loaded {} out of {} files'.format(i+1, len(self.all_files)))

        self.length = len(self.obs_features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.obs_features[idx], \
                self.lane_features[idx], \
                self.lane_features[idx].shape[0], \
                self.traj_labels[idx])


def collate_fn(batch):
    # batch is a list of tuples, so unzip to form lists of np-arrays.
    obs_features, lane_features, num_lanes, traj_labels = zip(*batch)

    # Stack into numpy arrays.
    N = len(obs_features)
    # N x 45
    obs_features = np.concatenate(obs_features, axis=0)
    # M x 400 (M >= N, M = sum(num_lanes))
    lane_features = np.concatenate(lane_features, axis=0)
    # list of N numbers
    num_lanes = np.asarray(num_lanes)
    # N x 30
    traj_labels = np.concatenate(traj_labels, axis=0)

    # Convert to torch tensors.
    return (torch.from_numpy(obs_features), \
            torch.from_numpy(lane_features), \
            torch.from_numpy(num_lanes)) ,\
           torch.from_numpy(traj_labels)

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
    return (torch.from_numpy(obs_features).view(N, 45), \
            torch.from_numpy(lane_features_padded).view(N, max_num_laneseq, 400), \
            torch.from_numpy(num_lanes).float()),\
           torch.from_numpy(traj_labels).view(N, 30)


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
    labels = np.zeros((N, 30))
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
        for j, point in enumerate(features[i][-30:-15]):
            labels[i-num_skipped, j], labels[i-num_skipped, j+15] =\
                world_coord_to_relative_coord(point, features[i][-30])
        # sanity check:
        if labels[i-num_skipped, 0] != 0.0 or labels[i-num_skipped, 15] != 0.0:
            print ('Future trajectory isn\'t correctly normalized.')

    if num_skipped == 0:
        return padded_features, labels
    else:
        return padded_features[:-num_skipped, :], labels[:-num_skipped, :]

def batch_preprocess(X, y):
    idx_1 = np.arange(X.shape[0]).tolist()
    idx_2 = torch.argsort(X[:,-1], descending=True).tolist()
    return X[idx_2], y[idx_2]


'''
========================================================================
Model definition
========================================================================
'''
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

        # TODO(jiacheng): add attention mechanisms to focus on some lanes.
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
            nn.Linear(46, 30)
        )

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
        obs_fea_final = self.obs_feature_fc(obs_fea.float())
        fea_all = torch.cat((obs_fea_final, static_fea_final), 1)
        # N x 79

        # TODO(jiacheng): use RNN decoder to output traj
        # TODO(jiacheng): multi-modal prediction
        traj = self.traj_fc(fea_all)
        # N x 30

        return traj


    def forward_dataloader(self, X):
        obs_fea, lane_fea, num_laneseq = X
        num_laneseq = num_laneseq.tolist()

        # Process each lane-sequence with CNN-1d
        # M x (100 * 4)
        lane_fea = lane_fea.view(lane_fea.size(0), 100, 4)
        lane_fea = lane_fea.transpose(1, 2).float()
        # M x 4 x 100
        lane_fea = self.single_lane_cnn(lane_fea)
        # M x 25 x 8
        lane_fea = self.single_lane_maxpool(lane_fea)
        # M x 25 x 2
        lane_fea = lane_fea.view(lane_fea.size(0), -1)
        lane_fea = self.single_lane_dropout(lane_fea)
        embed_dim = lane_fea.size(1)
        # M x 50

        # Group the lane-sequences for each obstacle
        max_num_laneseq = max(num_laneseq)
        N = obs_fea.size(0)
        # N x max_num_laneseq x 50
        lane_fea_rnn = torch.zeros(N, max_num_laneseq, embed_dim).cuda()
        end_idx = np.cumsum(num_laneseq).tolist()
        start_idx = ([0] + end_idx)[:-1]
        start_end_idx = [
            (start, end) for start, end in zip(start_idx, end_idx)]
        # list of N tuples
        for i, (start, end) in enumerate(start_end_idx):
            lane_fea_rnn[i, :num_laneseq[i], :] = lane_fea[start:end, :]
        # lane_fea_rnn contains lane_fea with padded zeros.

        # Sort and pack for RNN
        num_laneseq = torch.from_numpy(np.asarray(num_laneseq)).cuda()
        idx_new = torch.argsort(num_laneseq, descending=True).tolist()
        # N x 45
        obs_fea = obs_fea[idx_new]
        # N x max_num_laneseq x 50
        lane_fea_rnn = lane_fea_rnn[idx_new]
        # N x 1
        num_laneseq = num_laneseq[idx_new]
        num_laneseq = num_laneseq.float()

        # Process each obstacle's lane-sequences with RNN
        static_fea = pack_padded_sequence(lane_fea_rnn, num_laneseq, batch_first=True)
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
        obs_fea_final = self.obs_feature_fc(obs_fea.float())
        fea_all = torch.cat((obs_fea_final, static_fea_final), 1)
        # N x 79

        # TODO(jiacheng): use RNN decoder to output traj
        # TODO(jiacheng): multi-modal prediction
        traj = self.traj_fc(fea_all)
        # N x 30

        return traj

    def forward_vanilla(self, X):
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
        #TODO(jiacheng): elaborate on this (e.g. consider final displacement error, etc.)
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true.float())

    def loss_info(self, y_pred, y_true):
        return
