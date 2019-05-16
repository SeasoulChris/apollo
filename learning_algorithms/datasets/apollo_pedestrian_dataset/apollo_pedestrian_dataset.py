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

import data_for_learning_pb2
from data_for_learning_pb2 import *
from learning_algorithms.utilities.IO_utils import *


def LoadDataForLearning(filepath):
    list_of_data_for_learning = data_for_learning_pb2.ListDataForLearning()
    with open(filepath, 'rb') as file_in:
        list_of_data_for_learning.ParseFromString(file_in.read())
    return list_of_data_for_learning.data_for_learning


class ApolloPedestrianDataset(Dataset):
    def __init__(self, data_dir, obs_len=21, pred_len=40, threshold_dist_to_adc=30.0,
                 threshold_discontinuity=0.25, verbose=False):
        all_file_paths = GetListOfFiles(data_dir)
        seq_len = obs_len + pred_len
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

        # Go through every pedestrian:
        #   a. remove those that are too far away
        #   b. segment into different ped_ids at timestamp discontinuities.
        new_ped_id_to_traj_data = dict()
        ped_tracking_length = []
        for ped_id, ped_val in self.ped_id_to_traj_data.items():
            clean_ped_val = []
            clean_ped_timestamp = []
            # a:
            for ped_pt in ped_val:
                if ped_pt[3] < threshold_dist_to_adc:
                    clean_ped_val.append(ped_pt)
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
        

        if verbose:
            print ('Total number of usable pedestrians: {}'.format(len(self.ped_id_to_traj_data)))
            print ('Number of data with tracking length >= seq_len = {}'.format(
                       np.sum((ped_tracking_length - pred_len - obs_len) >= 0)))
            print ('Average tracking length = {}'.format(np.average(ped_tracking_length)))
            print ('Median tracking length = {}'.format(np.median(ped_tracking_length)))
            print ('Max tracking length = {}'.format(np.max(ped_tracking_length)))
            print ('Min tracking length = {}'.format(np.min(ped_tracking_length)))
            print ('Standard deviation of tracking length = {}'.format(np.std(ped_tracking_length)))

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx


if __name__ == "__main__":
    trial_dataset = ApolloPedestrianDataset('/home/jiacheng/work/apollo/data/all_pedestrian_data/features', verbose=True)
