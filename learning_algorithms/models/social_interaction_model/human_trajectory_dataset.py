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

'''Read a file that contains training data (helper function).

    Read files that contain training data line by line. Within each line,
    use "delim" as the deliminator. It will return a 2d-nparray that contains
    all the training data.
'''
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class HumanTrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 min_ped=0, delim='\t', extra_sample=0):
        all_file_paths = os.listdir(data_dir)
        all_file_paths = [os.path.join(data_dir, _path) for _path in all_file_paths]
        seq_len = obs_len + pred_len
        augmented_seq_len = (seq_len - 1) * (extra_sample + 1) + 1
        num_peds_in_scene = []
        self.extra_sample = int(extra_sample)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.scene_list = []
        self.scene_rel_list = []
        self.scene_timestamp_mask = []
        self.scene_is_predictable_list = []

        # Go through all the files that contain data
        for path in all_file_paths:
            data = read_file(path, delim)
            
            # Organize the data in the following way:
            #   All obstacles belonging to the same timestamp are clustered
            #   together; timestamp is sorted.
            # frame_data is a list of frames, with each frame containing all
            # data-points at that time-stamp.
            time_stamps = np.unique(data[:, 0]).tolist()
            frame_data = []
            for time_stamp in time_stamps:
                frame_data.append(data[data[:, 0] == time_stamp, :])
            num_sequences = int(math.ceil((len(time_stamps) - seq_len + 1) / skip))

            # A scene is defined as:
            #   A stream of frames lasting seq_len frames. During training, the
            #   interactions among all obstacles in the same scene are considered.
            # Go through every scene:
            for time_stamp_idx in range(0, num_sequences * skip + 1, skip):
                curr_scene_data = np.concatenate(
                    frame_data[time_stamp_idx:time_stamp_idx+seq_len], axis=0)
                peds_in_curr_scene_data = np.unique(curr_scene_data[:, 1])
                curr_scene = np.zeros(
                    (len(peds_in_curr_scene_data), augmented_seq_len, 2))
                curr_scene_rel = np.zeros(
                    (len(peds_in_curr_scene_data), augmented_seq_len, 2))
                curr_scene_timestamp_mask = np.zeros(
                    (len(peds_in_curr_scene_data), augmented_seq_len))
                curr_scene_is_predictable = np.zeros(
                    (len(peds_in_curr_scene_data), 1))
                # Go through every obstacle in the current scene:
                num_peds_considered = 0
                for i, ped_id in enumerate(peds_in_curr_scene_data):
                    curr_ped = curr_scene_data[curr_scene_data[:, 1] == ped_id, :]
                    curr_ped = np.around(curr_ped, decimals=4)
                    rel_time_begin = time_stamps.index(curr_ped[0, 0]) - time_stamp_idx
                    rel_time_end = time_stamps.index(curr_ped[-1, 0]) - time_stamp_idx + 1
                    # If this obstacle doesn't have enough number of frames,
                    # mark it as non-predictable. Vice versa.
                    curr_ped_is_predictable = False
                    if rel_time_end == seq_len and rel_time_begin < 5:
                        curr_ped_is_predictable = True

                    # Augment the data, if needed.
                    rel_time_begin = (self.extra_sample + 1) * rel_time_begin
                    rel_time_end = (rel_time_end - 1) * (self.extra_sample + 1) + 1
                    curr_ped = curr_ped[:, 2:]
                    curr_ped_aug = np.zeros(((curr_ped.shape[0] - 1) * (self.extra_sample + 1) + 1, 2))
                    for j in range(curr_ped.shape[0]-1):
                        xy_diff = curr_ped[j+1, :] - curr_ped[j, :]
                        for k in range(self.extra_sample+1):
                            curr_ped_aug[j*(self.extra_sample+1)+k, :] = curr_ped[j, :] + xy_diff*k/(self.extra_sample+1)
                    curr_ped_aug[curr_ped_aug.shape[0]-1, :] = curr_ped[curr_ped.shape[0]-1, :]
                    curr_ped = curr_ped_aug

                    # Get the coordinates of positions and make them relative.
                    # (relative position contains 1 fewer data-point, because, for
                    #  example, if there are 12 time-stamps, there will only be
                    #  11 intervals -- 11 relative displacements. The zeroth one
                    #  is set to 0.0 by default)
                    curr_ped_rel = np.zeros(curr_ped.shape)
                    curr_ped_timestamp_mask = np.ones((1, rel_time_end - rel_time_begin))
                    curr_ped_rel[1:, :] = curr_ped[1:, :] - curr_ped[:-1, :]
                    # Update into curr_scene matrix.
                    curr_scene[i, rel_time_begin:rel_time_end, :] = curr_ped
                    curr_scene_rel[i, rel_time_begin:rel_time_end, :] = curr_ped_rel
                    curr_scene_timestamp_mask[i, rel_time_begin:rel_time_end] = curr_ped_timestamp_mask
                    curr_scene_is_predictable[i] = curr_ped_is_predictable
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_scene.append(num_peds_considered)
                    self.scene_list.append(curr_scene)
                    self.scene_rel_list.append(curr_scene_rel)
                    self.scene_timestamp_mask.append(curr_scene_timestamp_mask)
                    self.scene_is_predictable_list.append(curr_scene_is_predictable)
        self.num_scene = len(self.scene_list)

    def __len__(self):
        return self.num_scene

    def __getitem__(self, idx):
        aug_obs_len = (self.obs_len - 1) * (self.extra_sample + 1) + 1
        out = (self.scene_list[idx][:, 0:aug_obs_len, :],
               self.scene_rel_list[idx][:, 0:aug_obs_len, :],
               self.scene_list[idx][:, aug_obs_len:, :],
               self.scene_rel_list[idx][:, aug_obs_len:, :],
               self.scene_timestamp_mask[idx][:, 0:aug_obs_len],
               self.scene_is_predictable_list[idx])
        # TODO(jiacheng): may need some preprocessing such as adding Gaussian noise, etc.
        return out


def collate_scenes(batch):
    # batch is a list of tuples
    # unzip to form list of np-arrays
    past_traj, past_traj_rel, pred_traj, pred_traj_rel, past_traj_timestamp_mask, is_predictable = zip(*batch)

    same_scene_mask = [scene.shape[0] for scene in past_traj]
    past_traj = np.concatenate(past_traj)
    past_traj_rel = np.concatenate(past_traj_rel)
    pred_traj = np.concatenate(pred_traj)
    pred_traj_rel = np.concatenate(pred_traj_rel)
    past_traj_timestamp_mask = np.concatenate(past_traj_timestamp_mask)
    is_predictable = np.concatenate(is_predictable)

    same_scene_mask = [np.ones((length, 1))*i for i, length in enumerate(same_scene_mask)]
    same_scene_mask = np.concatenate(same_scene_mask)

    return (torch.from_numpy(past_traj), torch.from_numpy(past_traj_rel), \
            torch.from_numpy(past_traj_timestamp_mask), torch.from_numpy(is_predictable), torch.from_numpy(same_scene_mask)),\
           (torch.from_numpy(pred_traj), torch.from_numpy(pred_traj_rel),\
            torch.from_numpy(is_predictable))
