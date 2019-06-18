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

obs_feature_size = 180
single_lane_feature_size = 600
past_lane_feature_size = 200
future_lane_feature_size = 400

def LoadDataForLearning(filepath):
    list_of_data_for_learning = \
        learning_algorithms.datasets.apollo_pedestrian_dataset.data_for_learning_pb2.ListDataForLearning()
    try:
        with open(filepath, 'rb') as file_in:
            list_of_data_for_learning.ParseFromString(file_in.read())
        return list_of_data_for_learning.data_for_learning
    except:
        return None


def CalculateDistanceAndAngle(curve, point):
    '''Calculate the distance and angle-diff between the point and given curve
        - curve: a list of tuples (x, y)
        - point: a tuple (x, y)

        return dist, angle-diff
    '''
    min_dist = None
    for i in range(len(curve)):
        distance = np.sqrt((point[0] - curve[len(curve)-1-i][0])**2 + \
                           (point[1] - curve[len(curve)-1-i][1])**2)
        if min_dist is None:
            min_dist = distance
        else:
            if distance > min_dist:
                break
            else:
                min_dist = distance

    return (min_dist, 0)

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
        label_path = os.path.join(dir_path, 'visited_lane_segment.npy')
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

            # 2. Deserialize the lane_graph, and remove data-points with incorrect sizes.
            lane_graph = []     # list of lane_sequences, each lane_sequence is represented by a frozenset.
            lane_sequence = None
            for lane_seg_id in serial_lane_graph:
                if lane_seg_id == '|':
                    if lane_sequence is not None:
                        lane_graph.append(frozenset(lane_sequence))
                    lane_sequence = []
                else:
                    lane_sequence.append(lane_seg_id)
            num_lane_sequence = len(lane_graph)
            # Remove corrupted data points.
            if (len(features_for_learning) - obs_feature_size) % single_lane_feature_size != 0 or \
               (len(features_for_learning) - obs_feature_size) / single_lane_feature_size != num_lane_sequence:
                num_dirty_data_point += 1
                continue
            # Get a set of unique lane-sequences only
            unique_lane_sequence_set = set()
            for lane_sequence in lane_graph:
                unique_lane_sequence_set.add(lane_sequence)
            if len(unique_lane_sequence_set) == 1:
                num_dirty_data_point += 1
                continue

            # 3. Extract the features of obstacle's historical distances/angles w.r.t. the lane-sequences.
            #    a. First, calculate the historical distances/angles w.r.t. all lane-sequences.
            list_of_backward_lane_points = []    # Contains num_lane_sequence lists, while each list of tuples represents a lane-curve.
            obs_past_history = features_for_learning[:obs_feature_size]
            for i in range(num_lane_sequence):
                lane_points_l = features_for_learning[obs_feature_size+i*single_lane_feature_size:\
                                                      obs_feature_size+(i+1)*single_lane_feature_size:4]
                lane_points_s = features_for_learning[obs_feature_size+i*single_lane_feature_size+1:\
                                                      obs_feature_size+(i+1)*single_lane_feature_size:4]
                curr_backward_lane_points = list(zip(lane_points_l, lane_points_s))
                curr_backward_lane_points = curr_backward_lane_points[0:int(past_lane_feature_size/4)]
                list_of_backward_lane_points.append(curr_backward_lane_points)
            list_of_dists_and_angles = []   # Contains num_lane_sequence lists, while each list of tuples represents historical dist and angle.
            for lane_points in list_of_backward_lane_points:
                curr_historical_dist_and_angle = []
                for obs_hist_idx in range(int(obs_feature_size/9)):
                    # If no history, then exit.
                    if obs_past_history[obs_hist_idx*9] == 0:
                        break
                    curr_historical_dist_and_angle.append(CalculateDistanceAndAngle(\
                        lane_points, (obs_past_history[obs_hist_idx*9+2], obs_past_history[obs_hist_idx*9+1])))
                list_of_dists_and_angles.append(curr_historical_dist_and_angle)
            #    b. Second, if multiple backward lane merges into one, then remove those that have exactly the same avg displacement.
            list_of_avg_displacement = []
            for curr_historical_dist_and_angle in list_of_dists_and_angles:
                list_of_avg_displacement.append(sum(curr_dist_and_angle[0] for curr_dist_and_angle in curr_historical_dist_and_angle))
            valid_lane_sequence_ids = []
            for unique_lane_sequence in unique_lane_sequence_set:
                avg_disp_set = set()
                for i in range(len(lane_graph)):
                    lane_sequence = lane_graph[i]
                    if lane_sequence == unique_lane_sequence:
                        if list_of_avg_displacement[i] not in avg_disp_set:
                            avg_disp_set.add(list_of_avg_displacement[i])
                            valid_lane_sequence_ids.append(i)
            new_lane_graph = []
            new_lane_disp_and_angle = []
            for lane_seq_id in valid_lane_sequence_ids:
                new_lane_graph.append(lane_graph[lane_seq_id])
                new_lane_disp_and_angle.append(list_of_dists_and_angles[lane_seq_id])
            lane_graph = new_lane_graph
            # list_of_dists_and_angles = new_lane_disp_and_angle
            num_lane_sequence = len(lane_graph)

            # 4. Based on the lane graph, remove those jittering data points (data cleaning).
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
                num_dirty_data_point += 1
                continue
            is_dirty_data_point = False
            end_lane_sequences = stable_window_lane_sequences[-1]
            if len(end_lane_sequences) == 0:
                num_dirty_data_point += 1
                continue
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

            # 5. Extract the features of whether each lane is the self-lane or not.
            start_lane_sequences = set()
            start_lane_segment_id = visited_lane_segments[0][1]
            for i, lane_sequence in enumerate(lane_graph):
                if start_lane_segment_id in lane_sequence:
                    start_lane_sequences.add(i)
            self_lane_features = []
            for i in range(num_lane_sequence):
                if i in start_lane_sequences:
                    self_lane_features.append(1)
                else:
                    self_lane_features.append(0)

            # 6. Label whether the obstacle has stepped out of its original lane-sequence(s).
            has_stepped_out = 1
            for i in end_lane_sequences:
                if i in start_lane_sequences:
                    has_stepped_out = 0
            num_cutin_data_points += has_stepped_out

            # 7. Refactor the label into the format of [1, 1, 0, 0, 0] ... (assume there are five lane sequences)
            one_hot_encoding_label = []
            for i in range(num_lane_sequence):
                if i in end_lane_sequences:
                    one_hot_encoding_label.append(1)
                else:
                    one_hot_encoding_label.append(0)

            # Put everything together.
            curr_data_point = [num_lane_sequence] + features_for_learning[:obs_feature_size]
            for i in valid_lane_sequence_ids:
                curr_data_point += features_for_learning[obs_feature_size+i*single_lane_feature_size:\
                                                         obs_feature_size+(i+1)*single_lane_feature_size]
                past_distances = [-1 for j in range(int(obs_feature_size/9))]
                for j in range(len(list_of_dists_and_angles[i])):
                    past_distances[j] = list_of_dists_and_angles[i][j][0]
                curr_data_point += past_distances
            curr_data_point += self_lane_features + one_hot_encoding_label + [has_stepped_out]
            output_np_array.append(curr_data_point)

        # Save into a local file for training.
        try:
            num_usable_data_points = len(output_np_array)
            print ('Removed dirty data points: {}'.format(num_dirty_data_point))
            print ('Total usable data points: {}'.format(num_usable_data_points))
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
    def __init__(self, data_dir, is_lane_scanning=True, training_mode=True, \
                 cutin_augmentation_coeff=1):
        self.obstacle_hist_size = []
        self.obstacle_features = []
        self.backward_lane_points = []
        self.lane_features = []
        self.is_self_lane = []
        self.labels = []
        self.is_cutin = []
        total_num_cutin_data_pt = 0

        all_file_paths = GetListOfFiles(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path).tolist()
            for data_pt in file_content:
                curr_num_lane_sequence = int(data_pt[0])
                if len(data_pt) != curr_num_lane_sequence*(single_lane_feature_size+20+2)+obs_feature_size+2:
                    continue

                curr_obs_hist_size = np.sum(np.array(data_pt[1:obs_feature_size+1:9])) * np.ones((1, 1))

                curr_obs_feature = np.array(data_pt[1:obs_feature_size+1]).reshape((int(obs_feature_size/9), 9))
                curr_obs_feature = np.flip(curr_obs_feature, 0)
                new_curr_obs_feature = np.zeros((int(obs_feature_size/9), 9))
                new_curr_obs_feature[:int(curr_obs_hist_size[0][0]), :] = curr_obs_feature[-int(curr_obs_hist_size[0][0]):, :]
                curr_obs_feature = new_curr_obs_feature.reshape((1, obs_feature_size))
                # TODO(jiacheng): remove dirty obstacle-feature data-points.
                #   - Remove data-points with history that has positive s values.
                #   - Remove data-points with history that has big abrupt jumps.

                curr_lane_feature = np.array(data_pt[obs_feature_size+1:obs_feature_size+1+\
                                                     (single_lane_feature_size+20)*curr_num_lane_sequence])\
                                    .reshape((curr_num_lane_sequence, single_lane_feature_size+20))
                curr_backward_lane_points = curr_lane_feature[:, single_lane_feature_size:]
                curr_backward_lane_points = np.flip(curr_backward_lane_points, 1)
                new_curr_backward_lane_points = np.zeros((curr_num_lane_sequence, 20))
                new_curr_backward_lane_points[:int(curr_obs_hist_size[0][0]), :] = curr_backward_lane_points[-int(curr_obs_hist_size[0][0]):, :]
                curr_backward_lane_points = new_curr_backward_lane_points

                curr_lane_feature = curr_lane_feature[:, :single_lane_feature_size]

                curr_self_lane_feature = np.array(data_pt[-1-2*curr_num_lane_sequence:-1-curr_num_lane_sequence]).reshape((curr_num_lane_sequence, 1))

                curr_label = np.array(data_pt[-1-curr_num_lane_sequence:-1])
                if np.sum(curr_label) == 0:
                    continue

                curr_is_cutin = data_pt[-1] * np.ones((1, 1))

                aug_times = 1
                if curr_is_cutin[0,0]:
                    aug_times = cutin_augmentation_coeff
                    total_num_cutin_data_pt += cutin_augmentation_coeff
                for k in range(aug_times):
                    if training_mode:
                        for i, lane_label in enumerate(curr_label):
                            if lane_label == 1:
                                self.obstacle_hist_size.append(curr_obs_hist_size)
                                self.obstacle_features.append(curr_obs_feature)
                                self.backward_lane_points.append(curr_backward_lane_points)
                                self.lane_features.append(curr_lane_feature)
                                self.is_self_lane.append(curr_self_lane_feature)
                                curr_lane_label = np.zeros((curr_num_lane_sequence, 1))
                                curr_lane_label[i, 0] = 1
                                self.labels.append(curr_lane_label)
                                self.is_cutin.append(curr_is_cutin)
                    else:
                        self.obstacle_hist_size.append(curr_obs_hist_size)
                        self.obstacle_features.append(curr_obs_feature)
                        self.backward_lane_points.append(curr_backward_lane_points)
                        self.lane_features.append(curr_lane_feature)
                        self.is_self_lane.append(curr_self_lane_feature)
                        self.labels.append(curr_label.reshape((curr_num_lane_sequence, 1)))
                        self.is_cutin.append(curr_is_cutin)

        self.total_num_data_pt = len(self.obstacle_features)
        print ('Total number of data points = {}'.format(self.total_num_data_pt))
        print ('Total number of cut-in data points = {}'.format(total_num_cutin_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        out = (self.obstacle_hist_size[idx], self.obstacle_features[idx], self.backward_lane_points[idx],\
               self.lane_features[idx], self.is_self_lane[idx], self.labels[idx], self.is_cutin[idx])
        return out


def collate_fn(batch):
    # batch is a list of tuples.
    # unzip to form lists of np-arrays.
    obs_hist_size, obs_features, backward_lane_points, lane_features, is_self_lane, labels, is_cutin = zip(*batch)

    same_obstacle_mask = [elem.shape[0] for elem in lane_features]
    obs_hist_size = np.concatenate(obs_hist_size)
    obs_features = np.concatenate(obs_features)
    backward_lane_points = np.concatenate(backward_lane_points)
    lane_features = np.concatenate(lane_features)
    is_self_lane = np.concatenate(is_self_lane)
    labels = np.concatenate(labels)
    is_cutin = np.concatenate(is_cutin)

    same_obstacle_mask = [np.ones((length, 1))*i for i, length in enumerate(same_obstacle_mask)]
    same_obstacle_mask = np.concatenate(same_obstacle_mask)

    return (torch.from_numpy(obs_hist_size), torch.from_numpy(obs_features), torch.from_numpy(backward_lane_points), \
            torch.from_numpy(lane_features), torch.from_numpy(is_self_lane), torch.from_numpy(same_obstacle_mask)), \
           (torch.from_numpy(labels), torch.from_numpy(is_cutin), torch.from_numpy(same_obstacle_mask))


if __name__ == '__main__':
    DataPreprocessing('/home/jiacheng/work/apollo/data/apollo_vehicle_regroad_data/test_data_preprocessing/features',\
                      '/home/jiacheng/work/apollo/data/apollo_vehicle_regroad_data/test_data_preprocessing/features')
