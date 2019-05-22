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
    try:
        with open(filepath, 'rb') as file_in:
            list_of_data_for_learning.ParseFromString(file_in.read())
        return list_of_data_for_learning.data_for_learning
    except:
        return None


def DataPreprocessing(feature_dir, label_dir, pred_len=3.0, stable_window=0.5):
    # Go through all the data_for_learning file, for each data-point, find
    # the corresponding label file, merge them.
    all_file_paths = GetListOfFiles(feature_dir)
    total_num_dirty_data_points = 0
    total_usable_data_points = 0
    total_cutin_data_points = 0

    file_count = 0
    for path in all_file_paths:
        file_count += 1
        # Load the label dict file.
        dir_path = os.path.dirname(path)
        label_path = os.path.join(dir_path, 'merged_visited_lane_segment.npy')
        label_path = label_path.replace('features', 'labels')
        if not os.path.isfile(label_path):
            continue
        dict_labels = np.load(label_path).item()
        # Load the feature for learning file.
        print ('============================================')
        print ('Reading file: {}. ({}/{})'.format(path, file_count, len(all_file_paths)))
        vector_data_for_learning = LoadDataForLearning(path)
        if vector_data_for_learning is None:
            print ('Failed to read file')
            continue

        num_dirty_data_point = 0
        num_cutin_data_points = 0
        output_np_array = []
        for data_for_learning in vector_data_for_learning:
            # Skip non-regroad-vehicle data:
            if data_for_learning.category != 'cruise':
                continue
            curr_data_point = []
            # 1. Find in the dict the corresponding visited lane segments.
            key = '{}@{:.3f}'.format(data_for_learning.id, data_for_learning.timestamp)
            if key not in dict_labels:
                continue
            features_for_learning = data_for_learning.features_for_learning
            serial_lane_graph = data_for_learning.string_features_for_learning
            serial_lane_graph.append('|')
            visited_lane_segments = dict_labels[key]

            # 2. Deserialize the lane_graph.
            lane_graph = []
            lane_sequence = None
            for lane_seg_id in serial_lane_graph:
                if lane_seg_id == '|':
                    if lane_sequence is not None:
                        lane_graph.append(lane_sequence)
                    lane_sequence = set()
                else:
                    lane_sequence.add(lane_seg_id)

            # 3. Based on the lane graph, remove those jittering data points (data cleaning).
            stable_window_lane_sequences = []
            for element in visited_lane_segments:
                timestamp = element[0]
                lane_seg_id = element[1]
                if timestamp > pred_len + 0.05:
                    break
                if timestamp > pred_len - stable_window + 0.05:
                    lane_seq_set = set()
                    for i, lane_sequence in enumerate(lane_graph):
                        if lane_seg_id in lane_sequence:
                            lane_seq_set.add(i)
                    stable_window_lane_sequences.append(lane_seq_set)
            if len(stable_window_lane_sequences) != int(stable_window/0.1):
                continue
            is_dirty_data_point = False
            end_lane_sequences = stable_window_lane_sequences[-1]
            for i in end_lane_sequences:
                for j in stable_window_lane_sequences:
                    if i not in j:
                        is_dirty_data_point = True
                        break
                if is_dirty_data_point:
                    break
            if is_dirty_data_point:
                num_dirty_data_point += 1
                continue

            # 4. Label whether the obstacle has stepped out of its original lane-sequence(s).
            start_lane_sequences = set()
            start_lane_segment_id = visited_lane_segments[0][1]
            for i, lane_sequence in enumerate(lane_graph):
                if start_lane_segment_id in lane_sequence:
                    start_lane_sequences.add(i)
            has_stepped_out = 1
            for i in end_lane_sequences:
                if i in start_lane_sequences:
                    has_stepped_out = 0
            num_cutin_data_points += has_stepped_out

            # 5. Refactor the label into the format of [1, 1, 0, 0, 0] ... (assume there are five lane sequences)
            one_hot_encoding_label = []
            for i in range(len(lane_graph)):
                if i in end_lane_sequences:
                    one_hot_encoding_label.append(1)
                else:
                    one_hot_encoding_label.append(0)
            curr_data_point = [len(lane_graph)] + features_for_learning[:180+400*len(lane_graph)] + \
                              one_hot_encoding_label + [has_stepped_out]
            output_np_array.append(curr_data_point)

        # Save into a local file for training.
        try:
            num_usable_data_points = len(output_np_array)
            print ('Total usable data points: {}'.format(num_usable_data_points))
            print ('Removed dirty data points: {}'.format(num_dirty_data_point))
            print ('Total cut-in data points: {}'.format(num_cutin_data_points))
            output_np_array = np.array(output_np_array)
            np.save(path+'.training_data.npy', output_np_array)
            total_usable_data_points += num_usable_data_points
            total_num_dirty_data_points += num_dirty_data_point
            total_cutin_data_points += num_cutin_data_points
        except:
            print ('Failed to save output file.')

    print ('Removed {} dirty data points.'.format(total_num_dirty_data_points))
    print ('There are {} usable data points.'.format(total_usable_data_points))
    print ('There are {} cut-in data points.'.format(total_cutin_data_points))


class ApolloVehicleRegularRoadDataset(Dataset):
    def __init__(self, data_dir, is_lane_scanning=True):
        self.obstacle_features = []
        self.lane_features = []
        self.labels = []
        self.is_cutin = []

        all_file_paths = GetListOfFiles(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path).tolist()
            for data_pt in file_content:
                curr_num_lane_sequence = int(data_pt[0])
                curr_obs_feature = np.array(data_pt[1:181]).reshape((1, 180))
                curr_lane_feature = np.array(data_pt[181:181+400*curr_num_lane_sequence])
                                    .reshape((curr_num_lane_sequence, 400))
                curr_label = np.array(data_pt[-1-curr_num_lane_sequence:-1])
                curr_is_cutin = data_pt[-1] * np.ones((1, 1))

                if is_lane_scanning:
                    self.obstacle_features.append(curr_obs_feature)
                    for i, lane_label in enumerate(curr_label):
                        if lane_label == 1:
                            self.obstacle_features.append(curr_obs_feature)
                            self.lane_features.append(curr_lane_feature)
                            curr_lane_label = np.zeros((curr_num_lane_sequence, 1))
                            curr_lane_label[i, 0] = 1
                            self.labels.append(curr_lane_label)
                            self.is_cutin.append(curr_is_cutin)
                else:
                    # TODO(jiacheng): implement this for regular cruiseMLP model.
                    1 == 1

        self.total_num_data_pt = len(self.obstacle_features)

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        out = (self.obstacle_features[idx], self.lane_features[idx],
               self.labels[idx], self.is_cutin[idx])
        return out


def collate_fn(batch):
    # batch is a list of tuples.
    # unzip to form lists of np-arrays.
    obs_features, lane_features, labels, is_cutin = zip(*batch)

    same_obstacle_mask = [elem[0] for elem in lane_features]
    obs_features = np.concatenate(obstacle_features)
    lane_features = np.concatenate(lane_features)
    labels = np.concatenate(labels)
    is_cutin = np.concatenate(is_cutin)

    same_obstacle_mask = [np.ones((length, 1))*i for i, length in enumerate(same_obstacle_mask)]
    same_obstacle_mask = np.concatenate(same_obstacle_mask)

    return (torch.from_numpy(obs_features), torch.from_numpy(lane_features), torch.from_numpy(same_obstacle_mask)), \
           (torch.from_numpy(labels), torch.from_numpy(is_cutin), torch.from_numpy(same_obstacle_mask))


if __name__ == '__main__':
    DataPreprocessing('/home/jiacheng/work/apollo/data/vehicle_regroad_data/features-2019-05-16', 
                      '/home/jiacheng/work/apollo/data/apollo_vehicle_regroad_data/labels/')
