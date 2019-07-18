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

from collections import Counter
import cv2 as cv
import glob
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

import learning_algorithms.datasets.apollo_pedestrian_dataset.data_for_learning_pb2
from learning_algorithms.datasets.apollo_pedestrian_dataset.data_for_learning_pb2 import *
from learning_algorithms.utilities.IO_utils import *
from learning_algorithms.utilities.helper_utils import *


obs_feature_size = 180
single_lane_feature_size = 600
past_lane_feature_size = 200
future_lane_feature_size = 400


import scipy
from scipy.signal import filtfilt


class ApolloVehicleTrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.obs_hist_sizes = []
        self.obs_pos = []
        self.obs_pos_rel = []
        self.lane_feature = []
        self.future_traj = []
        self.future_traj_rel = []
        total_num_cutin_data_pt = 0

        all_file_paths = GetListOfFiles(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path).tolist()
            for data_pt in file_content:
                # Get number of lane-sequences info.
                curr_num_lane_sequence = int(data_pt[0])

                # Get the size of obstacle state history.
                curr_obs_hist_size = int(np.sum(np.array(data_pt[1:obs_feature_size+1:9])))
                if curr_obs_hist_size <= 1:
                    continue
                self.obs_hist_sizes.append(curr_obs_hist_size * np.ones((1, 1)))

                # Get the obstacle features (organized from past to present).
                # (if length not enough, then pad heading zeros)
                curr_obs_feature = np.array(data_pt[1:obs_feature_size+1]).reshape((int(obs_feature_size/9), 9))
                curr_obs_feature = np.flip(curr_obs_feature, 0)
                curr_obs_pos = np.zeros((1, int(obs_feature_size/9), 2))
                # (1 x max_obs_hist_size x 2)
                curr_obs_pos[0, -curr_obs_hist_size:, :] = curr_obs_feature[-curr_obs_hist_size:, 1:3]
                self.obs_pos.append(curr_obs_pos)
                curr_obs_pos_rel =  np.zeros((1, int(obs_feature_size/9), 2))
                curr_obs_pos_rel[0, -curr_obs_hist_size+1:, :] = \
                    curr_obs_pos[0, -curr_obs_hist_size+1:, :] - curr_obs_pos[0, -curr_obs_hist_size:-1, :]
                self.obs_pos_rel.append(curr_obs_pos_rel)

                # Get the lane features.
                # (curr_num_lane_sequence x num_lane_pts x 4)
                curr_lane_feature = np.array(data_pt[obs_feature_size+1:obs_feature_size+1+\
                                                     (single_lane_feature_size)*curr_num_lane_sequence])\
                                    .reshape((curr_num_lane_sequence, int(single_lane_feature_size/4), 4))
                curr_lane_feature[:, :, [0, 1]] = curr_lane_feature[:, :, [1, 0]]
                # Remove too close lane-points.
                curr_lane_feature = np.concatenate(\
                    (curr_lane_feature[:, :49, :], curr_lane_feature[:, 51:, :]), axis=1)
                # The following part appends a beginning and an ending point.
                begin_pt = 2*curr_lane_feature[:, 0, :] - 1*curr_lane_feature[:, 1, :]
                begin_pt[:, 2] = curr_lane_feature[:, 0, 2]
                begin_pt[:, 3] = np.zeros((curr_num_lane_sequence))
                end_pt = 2*curr_lane_feature[:, -1, :] - 1*curr_lane_feature[:, -2, :]
                end_pt[:, 2] = curr_lane_feature[:, -1, 2]
                end_pt[:, 3] = np.zeros((curr_num_lane_sequence))
                curr_lane_feature = np.concatenate((begin_pt.reshape((curr_num_lane_sequence, 1, 4)),\
                    curr_lane_feature, end_pt.reshape((curr_num_lane_sequence, 1, 4))), axis=1)
                self.lane_feature.append(curr_lane_feature)

                # TODO(jiacheng): get the self-lane features.

                # Get the future trajectory label.
                curr_future_traj = np.array(data_pt[-91:-31]).reshape((2, 30))
                curr_future_traj = curr_future_traj.transpose()
                ref_world_coord = [curr_future_traj[0, 0], curr_future_traj[0, 1], data_pt[-31]]
                new_curr_future_traj = np.zeros((1, 30, 2))
                for i in range(30):
                    new_coord = world_coord_to_relative_coord(curr_future_traj[i, :], ref_world_coord)
                    new_curr_future_traj[0, i, 0] = new_coord[0]
                    new_curr_future_traj[0, i, 1] = new_coord[1]
                # (1 x 30 x 2)
                self.future_traj.append(new_curr_future_traj)
                curr_future_traj_rel = np.zeros((1, 29, 2))
                curr_future_traj_rel = new_curr_future_traj[:, 1:, :] - new_curr_future_traj[:, :-1, :]
                # (1 x 29 x 2)
                self.future_traj_rel.append(curr_future_traj_rel)

        self.total_num_data_pt = len(self.obs_pos)
        print ('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        out = (self.obs_hist_sizes[idx], self.obs_pos[idx], self.obs_pos_rel[idx], self.lane_feature[idx],\
               self.future_traj[idx], self.future_traj_rel[idx])
        return out


def collate_fn(batch):
    '''
    return:
        - obs_hist_size: N x 1
        - obs_pos: N x max_obs_hist_size x 2
        - obs_pos_rel: N x max_obs_hist_size x 2
        - lane_features: M x 152 x 4
        - same_obstacle_mask: M x 1

        - future_traj: N x 30 x 2
        - future_traj_rel: N x 29 x 2
        - is_predictable: N x 1
    '''
    # batch is a list of tuples.
    # unzip to form lists of np-arrays.
    obs_hist_size, obs_pos, obs_pos_rel, lane_features, future_traj, future_traj_rel = zip(*batch)

    same_obstacle_mask = [elem.shape[0] for elem in lane_features]
    obs_hist_size = np.concatenate(obs_hist_size)
    obs_pos = np.concatenate(obs_pos)
    obs_pos_rel = np.concatenate(obs_pos_rel)
    lane_features = np.concatenate(lane_features)
    future_traj = np.concatenate(future_traj)
    future_traj_rel = np.concatenate(future_traj_rel)

    same_obstacle_mask = [np.ones((length, 1))*i for i, length in enumerate(same_obstacle_mask)]
    same_obstacle_mask = np.concatenate(same_obstacle_mask)

    return (torch.from_numpy(obs_hist_size), torch.from_numpy(obs_pos), torch.from_numpy(obs_pos_rel), \
            torch.from_numpy(lane_features).float(), torch.from_numpy(same_obstacle_mask)), \
           (torch.from_numpy(future_traj), torch.from_numpy(future_traj_rel), torch.ones(obs_pos.shape[0], 1))
