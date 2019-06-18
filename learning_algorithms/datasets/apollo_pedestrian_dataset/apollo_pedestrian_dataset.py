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

import os
import math

import numpy as np
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


def LoadDataForLearning(filepath):
    list_of_data_for_learning = \
        learning_algorithms.datasets.apollo_pedestrian_dataset.data_for_learning_pb2.ListDataForLearning()
    with open(filepath, 'rb') as file_in:
        list_of_data_for_learning.ParseFromString(file_in.read())
    return list_of_data_for_learning.data_for_learning


class ApolloPedestrianDataset(Dataset):
    def __init__(self, data_dir, obs_len=21, pred_len=40, threshold_dist_to_adc=15.0,
                 threshold_discontinuity=0.25, verbose=False):
        all_file_paths = GetListOfFiles(data_dir)
        seq_len = obs_len + pred_len
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.ped_id_to_traj_data = dict()
        self.scene_list = []
        self.scene_rel_list = []
        self.scene_timestamp_mask = []
        self.scene_is_predictable_list = []

        # Go through all files, save pedestrian data by their IDs.
        for path in all_file_paths:
            vector_data_for_learning = LoadDataForLearning(path)
            for data_for_learning in vector_data_for_learning:
                # Skip non-pedestrian data.
                if data_for_learning.category != 'pedestrian':
                    continue
                ped_id = data_for_learning.id
                ped_timestamp = data_for_learning.timestamp
                ped_x = data_for_learning.features_for_learning[2]
                ped_y = data_for_learning.features_for_learning[3]
                adc_x = data_for_learning.features_for_learning[4]
                adc_y = data_for_learning.features_for_learning[5]
                # Store the value into dict (key is ped_id).
                if ped_id not in self.ped_id_to_traj_data:
                    self.ped_id_to_traj_data[ped_id] = []
                self.ped_id_to_traj_data[ped_id] = self.ped_id_to_traj_data[ped_id] + \
                    [(ped_timestamp, ped_x, ped_y, np.sqrt((ped_x-adc_x)**2 + (ped_y-adc_y)**2))]

        # Go through all peds, remove those that are too short to be usable.
        new_ped_id_to_traj_data = dict()
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            if len(ped_val) > pred_len + 1:
                new_ped_id_to_traj_data[ped_id] = ped_val
        self.ped_id_to_traj_data = new_ped_id_to_traj_data

        # Go through the dictionary, sort each pedestrian by timestamp.
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            sorted_ped_val = sorted(ped_val, key=lambda tup: tup[0])
            self.ped_id_to_traj_data[ped_id] = sorted_ped_val

        # Normalize large ped_x and ped_y values to smaller ones to avoid possible numerical issues.
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            normalized_ped_val = ped_val
            for i in range(len(normalized_ped_val)):
                normalized_ped_val[i] = (normalized_ped_val[i][0], normalized_ped_val[i][1] - normalized_ped_val[-1][1], \
                    normalized_ped_val[i][2] - normalized_ped_val[-1][2], normalized_ped_val[i][3])
            #print (normalized_ped_val)
            self.ped_id_to_traj_data[ped_id] = normalized_ped_val

        # Go through every pedestrian:
        #   a. remove those that are too far away from adc.
        #   b. segment into different ped_ids at timestamp discontinuities.
        new_ped_id_to_traj_data = dict()
        ped_tracking_length = []
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            clean_ped_val = []
            clean_ped_timestamp = []
            # a:
            for ped_pt in ped_val:
                if ped_pt[3] < threshold_dist_to_adc:
                    clean_ped_val.append([ped_pt[1], ped_pt[2]])
                    clean_ped_timestamp.append(ped_pt[0])
            if len(clean_ped_val) <= pred_len + 1:
                continue
            # b:
            clean_ped_timestamp = np.asarray(clean_ped_timestamp)
            seg_id = 0
            ped_discontinuity_idx = \
                np.argwhere(clean_ped_timestamp[1:] - clean_ped_timestamp[:-1] > threshold_discontinuity)
            ped_discontinuity_idx = [0] + (ped_discontinuity_idx.reshape(-1) + 1).tolist() + [len(clean_ped_val)]
            for i in range(len(ped_discontinuity_idx)-1):
                if ped_discontinuity_idx[i+1] - ped_discontinuity_idx[i] > pred_len + 1:
                    new_ped_id_to_traj_data[str(ped_id)+'_'+str(seg_id)] = clean_ped_val[ped_discontinuity_idx[i]:ped_discontinuity_idx[i+1]]
                    ped_tracking_length.append(ped_discontinuity_idx[i+1] - ped_discontinuity_idx[i])
                    seg_id += 1
        self.ped_id_to_traj_data = new_ped_id_to_traj_data
        ped_tracking_length = np.asarray(ped_tracking_length)

        # Go through all the pedestrians, and construct the scene_list, scene_rel_list,
        # scene_timestamp_mask, and scene_is_predictable_list, just like the public
        # HumanTrajectoryDataset.
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            if len(ped_val) <= seq_len:
                curr_scene = np.zeros((1, seq_len, 2))
                curr_scene_rel = np.zeros((1, seq_len, 2))
                curr_scene_timestamp_mask = np.zeros((1, seq_len))

                curr_scene[0, -len(ped_val): ,:] = np.asarray(ped_val)
                curr_scene_rel[0, -len(ped_val)+1: ,:] = np.asarray(ped_val)[1:, :] - np.asarray(ped_val)[:-1, :]
                curr_scene_timestamp_mask[0, -len(ped_val):] = np.ones((len(ped_val)))

                self.scene_list.append(curr_scene)
                self.scene_rel_list.append(curr_scene_rel)
                self.scene_timestamp_mask.append(curr_scene_timestamp_mask)
                self.scene_is_predictable_list.append(np.ones((1, 1)))
            else:
                for i in range(len(ped_val)-seq_len+1):
                    self.scene_list.append(np.asarray(ped_val[i:i+seq_len]).reshape((1, seq_len, 2)))
                    curr_scene_rel = np.zeros((1, seq_len, 2))
                    curr_scene_rel[0, 1: ,:] = np.asarray(ped_val[i:i+seq_len])[1:, :] - np.asarray(ped_val[i:i+seq_len])[:-1, :]
                    self.scene_rel_list.append(curr_scene_rel)
                    self.scene_timestamp_mask.append(np.ones((1, seq_len)))
                    self.scene_is_predictable_list.append(np.ones((1, 1)))
        self.num_scene = len(self.scene_list)

        if verbose:
            print ('Dataset size = {}'.format(self.num_scene))
            print ('Total number of usable pedestrians: {}'.format(len(self.ped_id_to_traj_data)))
            print ('Number of data with tracking length >= seq_len = {}'.format(
                       np.sum((ped_tracking_length - pred_len - obs_len) >= 0)))
            print ('Average tracking length = {}'.format(np.average(ped_tracking_length)))
            print ('Median tracking length = {}'.format(np.median(ped_tracking_length)))
            print ('Max tracking length = {}'.format(np.max(ped_tracking_length)))
            print ('Min tracking length = {}'.format(np.min(ped_tracking_length)))
            print ('Standard deviation of tracking length = {}'.format(np.std(ped_tracking_length)))

    def __len__(self):
        return self.num_scene

    def __getitem__(self, idx):
        out = (self.scene_list[idx][:, :self.obs_len, :],
               self.scene_rel_list[idx][:, :self.obs_len, :],
               self.scene_list[idx][:, self.obs_len:, :],
               self.scene_rel_list[idx][:, self.obs_len:, :],
               self.scene_timestamp_mask[idx][:, :self.obs_len],
               self.scene_is_predictable_list[idx])
        return out
