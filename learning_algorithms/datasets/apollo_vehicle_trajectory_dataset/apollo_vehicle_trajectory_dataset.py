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


obs_feature_size = 180
single_lane_feature_size = 600
past_lane_feature_size = 200
future_lane_feature_size = 400


import scipy
from scipy.signal import filtfilt


def point_to_idx(point_x, point_y):
    return (int(point_x/0.1 + 400), int(point_y/0.1 + 400))


def plot_img(future_pt, count, adjusted_future_pt=None):
    # black background
    img = np.zeros([1000, 800, 3], dtype=np.uint8)
    # draw boundaries
    cv.circle(img, (400, 400), 2, color=[255, 255, 255], thickness=4)
    cv.line(img, (0, 0), (799, 0), color=[255, 255, 255])
    cv.line(img, (799, 0), (799, 999), color=[255, 255, 255])
    cv.line(img, (0, 999), (0, 0), color=[255, 255, 255])
    cv.line(img, (799, 999), (0, 999), color=[255, 255, 255])
    for ts in range(future_pt.shape[0]):
        cv.circle(img, point_to_idx(future_pt[ts][0] - future_pt[0][0], future_pt[ts]
                                    [1] - future_pt[0][1]), radius=3, thickness=2, color=[0, 128, 128])
    if adjusted_future_pt is not None:
        for ts in range(adjusted_future_pt.shape[0]):
            cv.circle(img, point_to_idx(adjusted_future_pt[ts][0] - adjusted_future_pt[0][0], adjusted_future_pt[ts]
                                        [1] - adjusted_future_pt[0][1]), radius=3, thickness=2, color=[128, 0, 128])
    cv.imwrite('img={}.png'.format(count), cv.flip(cv.flip(img, 0), 1))


def LabelValid(feature_seq, pred_len=30):
    # 1. Only keep pred_len length
    if len(feature_seq) < pred_len:
        return None
    obs_pos = np.array([[feature[0], feature[1]]
                        for feature in feature_seq[:pred_len]])
    obs_pos = obs_pos - obs_pos[0, :]
    # 2. Get the scalar acceleration, and angular speed of all points.
    obs_vel = (obs_pos[1:, :] - obs_pos[:-1, :]) / 0.1
    linear_vel = np.linalg.norm(obs_vel, axis=1)
    linear_acc = (linear_vel[1:] - linear_vel[0:-1]) / 0.1
    angular_vel = np.sum(
        obs_vel[1:, :] * obs_vel[:-1, :], axis=1) / ((linear_vel[1:] * linear_vel[:-1]) + 1e-6)
    turning_ang = (np.arctan2(
        obs_vel[-1, 1], obs_vel[-1, 0]) - np.arctan2(obs_vel[0, 1], obs_vel[0, 0])) % (2*np.pi)
    turning_ang = turning_ang if turning_ang < np.pi else turning_ang-2*np.pi
    # 3. Filtered the extream values for acc and ang_vel.
    if np.max(np.abs(linear_acc)) > 50:
        return None
    if np.min(angular_vel) < 0.85:
        return None
    # Get the statistics of the cleaned labels, and do some re-balancing to
    # maintain roughly the same distribution as before.
    if -np.pi/6 <= turning_ang <= np.pi/6:
        area = (obs_pos[0, 0]*obs_pos[1, 1] + obs_pos[1, 0]*obs_pos[-1, 1] + obs_pos[-1, 0]*obs_pos[0, 1]
                - obs_pos[0, 0]*obs_pos[-1, 1] - obs_pos[1, 0]*obs_pos[0, 1] - obs_pos[-1, 0]*obs_pos[1, 1])
        if area/(np.linalg.norm(obs_pos[1, :] - obs_pos[0, :]) + 1e-6) >= 3:
            return 'change_lane'
        else:
            return 'straight'
    elif -np.pi/2 <= turning_ang < -np.pi/6:
        return 'right'
    elif np.pi/6 < turning_ang <= np.pi/2:
        return 'left'
    else:
        return 'uturn'


def SmoothFeatureSequence(feature_seq):
    """
    feature_seq: a sequence of tuples (x, y, v_heading, v, length, width, timestamp, acc)
    """
    x_coords = []
    y_coords = []
    smoothed_feature_seq = []
    start_x = feature_seq[0][0]
    start_y = feature_seq[0][1]

    for feature in feature_seq:
        x_coords.append(feature[0] - start_x)
        y_coords.append(feature[1] - start_y)

    b, a = scipy.signal.butter(8, 0.8)
    smooth_x_coords = filtfilt(b, a, x_coords, method="gust")
    smooth_y_coords = filtfilt(b, a, y_coords, method="gust")

    for i in range(len(feature_seq)):
        smoothed_feature = list(feature_seq[i])
        smoothed_feature[0] = smooth_x_coords[i] + start_x
        smoothed_feature[1] = smooth_y_coords[i] + start_y
        smoothed_feature_seq.append(tuple(smoothed_feature))

    return smoothed_feature_seq

def LabelProcessing(label_dir):
    label_dict_file_list = glob.glob(
        label_dir + '/**/future_status.npy', recursive=True)
    count = Counter()
    for label_dict_file in label_dict_file_list:
        label_dict = np.load(label_dict_file, allow_pickle=True).item()
        processed_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            pred_len = 30
            turn_type = LabelValid(feature_seq, pred_len)
            if turn_type:
                count[turn_type] += 1
                feature_seq = feature_seq[:pred_len]
                smoothed_feature_seq = SmoothFeatureSequence(feature_seq)
                processed_label_dict[key] = smoothed_feature_seq
                obs_pos = np.array([[feature[0], feature[1]]
                                    for feature in feature_seq])
                obs_pos = obs_pos - obs_pos[0, :]
                smoothed_obs_pos = np.array([[feature[0], feature[1]]
                                             for feature in smoothed_feature_seq])
                smoothed_obs_pos = smoothed_obs_pos - smoothed_obs_pos[0, :]
                # plot_img(obs_pos, idx, smoothed_obs_pos)
                # idx += 1
        print("Got " + str(len(processed_label_dict.keys())) +
              "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        # np.save(label_dict_name.replace('future_status.npy',
        #                                 'processed_label.npy'), processed_label_dict)


def LoadDataForLearning(filepath):
    list_of_data_for_learning = \
        learning_algorithms.datasets.apollo_pedestrian_dataset.data_for_learning_pb2.ListDataForLearning()
    try:
        with open(filepath, 'rb') as file_in:
            list_of_data_for_learning.ParseFromString(file_in.read())
        return list_of_data_for_learning.data_for_learning
    except:
        return None


def LabelCleaning(feature_dir, label_dir, pred_len=30):
    # From feature_dir, locate those labels of interests.
    label_dict_list = glob.glob(label_dir + '/**/cleaned_label.npy', recursive=True)
  
    # Go through all labels of interests, filter out those noisy ones and
    # only retain those clean ones.
    count = Counter()
    file_count = 0
    for label_dict_name in label_dict_list:
        file_count += 1
        print ('Processing {}/{}'.format(file_count, len(label_dict_list)))
        label_dict = np.load(label_dict_name).item()
        cleaned_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            # 1. Only keep pred_len length
            if len(feature_seq) < pred_len:
                continue
            obs_pos = np.array([[feature[0], feature[1]] for feature in feature_seq[:pred_len]])
            obs_pos = obs_pos - obs_pos[0, :]
            # 2. Get the scalar acceleration, and angular speed of all points.
            obs_vel = (obs_pos[1:, :] - obs_pos[:-1, :]) / 0.1
            linear_vel = np.linalg.norm(obs_vel, axis=1)
            linear_acc = (linear_vel[1:] - linear_vel[0:-1]) / 0.1
            angular_vel = np.sum(obs_vel[1:, :] * obs_vel[:-1, :], axis=1) / ((linear_vel[1:] * linear_vel[:-1]) + 1e-6)
            turning_ang = (np.arctan2(obs_vel[-1,1], obs_vel[-1,0]) - np.arctan2(obs_vel[0,1], obs_vel[0,0])) % (2*np.pi)
            turning_ang = turning_ang if turning_ang < np.pi else turning_ang-2*np.pi
            # 3. Filtered the extream values for acc and ang_vel.
            if np.max(np.abs(linear_acc)) > 50:
                continue
            if np.min(angular_vel) < 0.85:
                continue
            # plot_img(obs_pos, idx)
            # print(idx, key)
            # idx += 1

            # Get the statistics of the cleaned labels, and do some re-balancing to
            # maintain roughly the same distribution as before.
            if -np.pi/6 <= turning_ang <= np.pi/6:
                if np.min(angular_vel) < 0.9 or np.max(np.abs(linear_acc)) > 30:
                    continue
                area = (obs_pos[0,0]*obs_pos[1,1] + obs_pos[1,0]*obs_pos[-1,1] + obs_pos[-1,0]*obs_pos[0,1]
                       -obs_pos[0,0]*obs_pos[-1,1] - obs_pos[1,0]*obs_pos[0,1] - obs_pos[-1,0]*obs_pos[1,1])
                if area/(np.linalg.norm(obs_pos[1,:] - obs_pos[0,:]) + 1e-6) >= 3:
                    count['change_lane'] += 1
                else:
                    count['straight'] += 1
            elif -np.pi/2 <= turning_ang < -np.pi/6:
                count['right'] += 1
            elif np.pi/6 < turning_ang <= np.pi/2:
                if np.max(np.abs(linear_acc)) > 30:
                    continue
                count['left'] += 1
            else:
                count['uturn'] += 1
            cleaned_label_dict[key] = feature_seq[:pred_len]

        print("Got " + str(len(cleaned_label_dict.keys())) + "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        np.save(label_dict_name.replace('cleaned_label.npy', 'cleaner_label.npy'), cleaned_label_dict)
    print(count)
    return


class DataPreprocessor(object):
    def __init__(self, pred_len=3.0):
        self.pred_len = pred_len

    def load_numpy_dict(self, feature_file_path, label_dir='labels_future_trajectory', label_file='cleaner_label.npy'):
        '''Load the numpy dictionary file for the corresponding feature-file.
        '''
        dir_path = os.path.dirname(feature_file_path)
        label_path = os.path.join(dir_path, label_file)
        label_path = label_path.replace('features', label_dir)
        if not os.path.isfile(label_path):
            return None
        return np.load(label_path).item()

    def deserialize_and_construct_lane_graph(self, serial_lane_graph):
        '''Deserialize the serial_lane_graph and construct a lane_graph.

            - return: a lane-graph which is a list of lane_sequences, each lane_sequence is represented by a frozenset.
        '''
        if serial_lane_graph[-1] != '|':
            serial_lane_graph.append('|')

        lane_graph = []
        lane_sequence = None
        for lane_seg_id in serial_lane_graph:
            if lane_seg_id == '|':
                if lane_sequence is not None:
                    lane_graph.append(frozenset(lane_sequence))
                lane_sequence = []
            else:
                lane_sequence.append(lane_seg_id)
        return lane_graph

    def get_lane_sequence_id(self, lane_segment_id, lane_graph):
        lane_seq_set = set()
        for i, lane_sequence in enumerate(lane_graph):
            if lane_segment_id in lane_sequence:
                lane_seq_set.add(i)
        return lane_seq_set

    def preprocess_regroad_data(self, feature_dir):
        # Go through all the data_for_learning file, for each data-point, find
        # the corresponding label file, merge them.
        all_file_paths = GetListOfFiles(feature_dir)
        total_num_data_points = 0
        total_usable_data_points = 0
        total_cutin_data_points = 0

        file_count = 0
        for path in all_file_paths:
            file_count += 1
            print ('============================================')
            print ('Reading file: {}. ({}/{})'.format(path, file_count, len(all_file_paths)))
            # Load feature and label files.
            # Load the visited-lane-segments label dict files.
            visited_lane_segments_labels = self.load_numpy_dict(\
                path, label_dir='labels-visited-lane-segments', label_file='visited_lane_segment.npy')
            if visited_lane_segments_labels is None:
                print ('Failed to read visited_lane_segment label file.')
                continue
            # Load the future-trajectory label dict files.
            future_trajectory_labels = self.load_numpy_dict(\
                path, label_dir='labels_future_trajectory', label_file='cleaner_label.npy')
            if future_trajectory_labels is None:
                print ('Failed to read future_trajectory label file.')
                continue
            # Load the feature for learning file.
            vector_data_for_learning = LoadDataForLearning(path)
            if vector_data_for_learning is None:
                print ('Failed to read feature file.')
                continue

            # Go through the entries in this feature file.
            total_num_data_points += len(vector_data_for_learning)
            num_cutin_data_points = 0
            output_np_array = []
            for data_for_learning in vector_data_for_learning:
                # 0. Skip non-regroad-vehicle data:
                if data_for_learning.category != 'cruise':
                    continue
                curr_data_point = []

                # 1. Find in the dict the corresponding visited lane segments and future trajectory.
                key = '{}@{:.3f}'.format(data_for_learning.id, data_for_learning.timestamp)
                if key not in visited_lane_segments_labels or\
                   key not in future_trajectory_labels:
                    continue
                features_for_learning = data_for_learning.features_for_learning
                serial_lane_graph = data_for_learning.string_features_for_learning
                visited_lane_segments = visited_lane_segments_labels[key]
                future_trajectory = future_trajectory_labels[key]
                # Remove corrupted data points.
                if (len(features_for_learning) - obs_feature_size) % single_lane_feature_size != 0:
                    continue

                # 2. Deserialize the lane_graph, and remove data-points with incorrect sizes.
                # Note that lane_graph only contains lane_segment_ids starting from the vehicle's current position (not including past lane_segment_ids).
                lane_graph = self.deserialize_and_construct_lane_graph(serial_lane_graph)
                num_lane_sequence = len(lane_graph)
                # Remove corrupted data points.
                if (len(features_for_learning) - obs_feature_size) / single_lane_feature_size != num_lane_sequence:
                    continue
                # Get a set of unique future lane-sequences. (If two lanes merged in the past, they will belong to the same future_lane_sequence)
                unique_future_lane_sequence_set = set()
                for lane_sequence in lane_graph:
                    unique_future_lane_sequence_set.add(lane_sequence)

                # 3. Based on the lane graph, figure out what lane-sequence the end-point is in.
                end_lane_sequences = None
                for element in visited_lane_segments:
                    timestamp = element[0]
                    lane_seg_id = element[1]
                    if timestamp > self.pred_len + 0.05:
                        break
                    end_lane_sequences = self.get_lane_sequence_id(lane_seg_id, lane_graph)

                # 4. Extract the features of whether each lane is the self-lane or not.
                start_lane_segment_id = visited_lane_segments[0][1]
                start_lane_sequences = self.get_lane_sequence_id(start_lane_segment_id, lane_graph)
                self_lane_features = []
                for i in range(num_lane_sequence):
                    if i in start_lane_sequences:
                        self_lane_features.append(1)
                    else:
                        self_lane_features.append(0)

                # 5. Label whether the obstacle has stepped out of its original lane-sequence(s).
                has_stepped_out = 1
                for i in end_lane_sequences:
                    if i in start_lane_sequences:
                        has_stepped_out = 0
                num_cutin_data_points += has_stepped_out

                # 6. Put everything together.
                #   a. indicate the number of lane-sequences.
                curr_data_point = [num_lane_sequence]
                #   b. include the obstacle historical states features.
                curr_data_point += features_for_learning[:obs_feature_size]
                #   c. include the lane_features for each lane.
                for i in range(num_lane_sequence):
                    curr_data_point += features_for_learning[obs_feature_size+i*single_lane_feature_size:\
                                                             obs_feature_size+(i+1)*single_lane_feature_size]
                #   d. add whether it's self-lane feature.
                curr_data_point += self_lane_features
                #   e. add the unique future lane info.
                #   TODO(jiacheng): implement the above one.
                #   f. add trajectory labels.
                for i, traj in enumerate(zip(*future_trajectory)):
                    if i >= 3:
                        break
                    curr_data_point += list(traj)
                #   g. add whether it's cut-in labels.
                curr_data_point += [has_stepped_out]
                output_np_array.append(curr_data_point)

            # Save into a local file for training.
            try:
                num_usable_data_points = len(output_np_array)
                print ('Total usable data points: {} out of {}.'.format(\
                    num_usable_data_points, len(vector_data_for_learning)))
                print ('Total cut-in data points: {}, which is {}%.'.format(\
                    num_cutin_data_points, num_cutin_data_points/num_usable_data_points*100))
                output_np_array = np.array(output_np_array)
                np.save(path+'.training_data.npy', output_np_array)
                total_usable_data_points += num_usable_data_points
                total_cutin_data_points += num_cutin_data_points
            except:
                print ('Failed to save output file.')

        print ('There are {} usable data points out of {}.'.format(\
            total_usable_data_points, total_num_data_points))
        print ('There are {} cut-in data points, which is {}% of the total data points.'.format(\
            total_cutin_data_points, total_cutin_data_points/total_usable_data_points*100))


class ApolloVehicleTrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.obs_hist_sizes = []
        self.obs_pos = []
        self.obs_pos_rel = []
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
                curr_obs_pos = np.zeros(1, (int(obs_feature_size/9), 2))
                # (1 x max_obs_hist_size x 2)
                curr_obs_pos[0, -curr_obs_hist_size:, :] = curr_obs_feature[-curr_obs_hist_size:, 1:3]
                self.obs_pos.append(curr_obs_pos)
                curr_obs_pos_rel =  np.zeros(1, (int(obs_feature_size/9), 2))
                curr_obs_pos_rel[0, -curr_obs_hist_size+1:, :] = \
                    curr_obs_pos[0, -curr_obs_hist_size+1:, :] - curr_obs_pos[0, -curr_obs_hist_size:-1, :]
                self.obs_pos_rel.append(curr_obs_pos_rel)

                # TODO(jiacheng): get the lane features.
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

        self.total_num_data_pt = len(self.obstacle_features)
        print ('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        out = (self.obs_hist_sizes[idx], self.obs_pos[idx], self.obs_pos_rel[idx], \
               self.future_traj[idx], self.future_traj_rel[idx])
        return out


if __name__ == '__main__':
    LabelProcessing('/data/labels-future-points/')
    # LabelCleaning('test', '/home/jiacheng/work/apollo/data/apollo_vehicle_trajectory_data/labels-future-points-clean')

    # data_preprocessor = DataPreprocessor()
    # data_preprocessor.preprocess_regroad_data('/home/jiacheng/work/apollo/data/vehicle_regroad_dataset/features')
