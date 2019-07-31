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
from torchvision import transforms

from learning_algorithms.prediction.data_preprocessing.map_feature.online_mapping import ObstacleMapping
import learning_algorithms.prediction.datasets.apollo_pedestrian_dataset.data_for_learning_pb2
from learning_algorithms.prediction.datasets.apollo_pedestrian_dataset.data_for_learning_pb2 import *
from learning_algorithms.utilities.IO_utils import *
from learning_algorithms.utilities.helper_utils import *


obs_hist_size = 20
obs_unit_feature_size = 40 + 9
obs_feature_size = obs_hist_size * obs_unit_feature_size
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


class DataPreprocessor(object):
    def __init__(self, pred_len=3.0):
        self.pred_len = pred_len

    def load_numpy_dict(self, feature_file_path, label_dir='labels_future_trajectory', label_file='future_status_clean.npy'):
        '''Load the numpy dictionary file for the corresponding feature-file.
        '''
        dir_path = os.path.dirname(feature_file_path)
        label_path = os.path.join(dir_path, label_file)
        label_path = label_path.replace('features', label_dir)
        if label_path.find('sunnyvale_with_two_offices') != -1:
            label_path = label_path.replace('sunnyvale_with_two_offices', 'sunnyvale')
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

    def preprocess_data(self, feature_dir, involve_all_relevant_data=False):
        '''
        params:
            - feature_dir: the directory containing all data_for_learn
            - involve_all_relevant_data:
                - if False, then only include those data points with clean labels.
                - if True, then as long as the data point occurs at a timestamp
                  at which there is a clean label, then this data point, regardless
                  of whether itself contains clean label, will be put into the train-data.
        '''
        # Go through all the data_for_learning file, for each data-point, find
        # the corresponding label file, merge them.
        all_file_paths = GetListOfFiles(feature_dir)
        total_num_data_points = 0
        total_usable_data_points = 0

        file_count = 0
        for path in all_file_paths:
            file_count += 1
            print ('============================================')
            print ('Reading file: {}. ({}/{})'.format(path, file_count, len(all_file_paths)))

            # Load the future-trajectory label dict files.
            future_trajectory_labels = self.load_numpy_dict(\
                path, label_dir='labels_future_trajectory', label_file='future_status.npy')
            if future_trajectory_labels is None:
                print ('Failed to read future_trajectory label file.')
                continue
            # Load the feature for learning file.
            vector_data_for_learning = LoadDataForLearning(path)
            if vector_data_for_learning is None:
                print ('Failed to read feature file.')
                continue

            # If involve_all_relevant_data, then needs to go through all the
            # clean labels, find their timestamps, and record them.
            # Later, all data-point occuring at such timestamps, regardless of
            # whether they have corresponding clean labels, should go into train-data.
            key_ts_to_data_pt = dict()
            if involve_all_relevant_data:
                for key in future_trajectory_labels.keys():
                    key_ts = key.split('@')[1]
                    key_ts_to_data_pt[key_ts] = []

            # Go through the entries in this feature file.
            total_num_data_points += len(vector_data_for_learning)
            output_np_array = []
            for data_for_learning in vector_data_for_learning:
                curr_data_point = []
                # 0. Skip non-vehicle data:
                if data_for_learning.category != 'vehicle_cruise' and \
                   data_for_learning.category != 'vehicle_junction':
                    continue

                # 1. Find in the dict the corresponding future trajectory.
                future_trajectory = None
                key = '{}@{:.3f}'.format(data_for_learning.id, data_for_learning.timestamp)
                if involve_all_relevant_data:
                    if key.split('@')[1] not in key_ts_to_data_pt.keys():
                        continue
                else:
                    if key not in future_trajectory_labels:
                        continue
                if key in future_trajectory_labels:
                    future_trajectory = future_trajectory_labels[key]
                features_for_learning = data_for_learning.features_for_learning
                serial_lane_graph = data_for_learning.string_features_for_learning
                # Remove corrupted data points.
                if (len(features_for_learning) - obs_feature_size) % single_lane_feature_size != 0:
                    continue
                # Remove data points with zero lane-sequence:
                if (len(features_for_learning) == obs_feature_size):
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

                # 3. Put everything together.
                #   a. indicate the number of lane-sequences.
                curr_data_point = [num_lane_sequence]
                #   b. include the obstacle historical states features.
                curr_data_point += features_for_learning[:obs_feature_size]
                #   c. include the lane_features for each lane.
                for i in range(num_lane_sequence):
                    curr_data_point += features_for_learning[obs_feature_size+i*single_lane_feature_size:\
                                                             obs_feature_size+(i+1)*single_lane_feature_size]
                #   d. include the future_status labels.
                if future_trajectory is not None:
                    for i, traj in enumerate(zip(*future_trajectory)):
                        # (only use pos_x, pos_y, and vel_heading)
                        if i >= 3:
                            break
                        curr_data_point += list(traj)

                # 4. Update into the output_np_array.
                if involve_all_relevant_data:
                    key_ts_to_data_pt[key.split('@')[1]] = \
                        key_ts_to_data_pt[key.split('@')[1]].append(curr_data_point)
                else:
                    output_np_array.append([curr_data_point])

            if involve_all_relevant_data:
                # Every scene is a list of data_points, with every data_point
                # being a list of features & labels (optional).
                for _, scene in key_ts_to_data_pt.items():
                    output_np_array.append(scene)

            # Save into a local file for training.
            try:
                num_usable_data_points = len(output_np_array)
                print ('Total usable data points: {} out of {}.'.format(\
                    num_usable_data_points, len(vector_data_for_learning)))
                output_np_array = np.array(output_np_array)
                np.save(path+'.training_data.npy', output_np_array)
                total_usable_data_points += num_usable_data_points
            except:
                print ('Failed to save output file.')

        print ('There are {} usable data points out of {}.'.format(\
            total_usable_data_points, total_num_data_points))


class ApolloVehicleTrajectoryDataset(Dataset):
    def __init__(self, data_dir, img_mode=False):
        self.img_mode = img_mode
        self.img_transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

        self.obs_hist_sizes = []
        self.obs_pos = []
        self.obs_pos_rel = []
        self.obs_polygons = []

        self.lane_feature = []

        self.future_traj = []
        self.future_traj_rel = []

        self.is_predictable = []
        self.map_region = []

        self.start_idx = []
        self.end_idx = []

        self.reference_world_coord = []
        total_num_cutin_data_pt = 0
        accumulated_data_pt = 0

        # TODO(Hongyi): add the drawing class here.
        self.base_map = {"sunnyvale":cv.imread("sunnyvale_with_two_offices.png"), "san_mateo":cv.imread("san_mateo.png")}

        all_file_paths = GetListOfFiles(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path, allow_pickle=True).tolist()
            for scene in file_content:
                self.start_idx.append(accumulated_data_pt)
                for data_pt in scene:
                    accumulated_data_pt += 1
                    if file_path.find("sunnyvale")!=-1:
                        self.map_region.append("sunnyvale")
                    elif file_path.find("san_mateo")!=-1:
                        self.map_region.append("san_mateo")
                    else:
                        self.map_region.append("unknown")

                    # Get number of lane-sequences info.
                    curr_num_lane_sequence = int(data_pt[0])

                    # Get the size of obstacle state history.
                    curr_obs_hist_size = int(np.sum(np.array(data_pt[1:obs_feature_size+1:obs_unit_feature_size])))
                    if curr_obs_hist_size <= 1:
                        accumulated_data_pt -= 1
                        continue
                    self.obs_hist_sizes.append(curr_obs_hist_size * np.ones((1, 1)))

                    # Get the obstacle position features (organized from past to present).
                    # (if length not enough, then pad heading zeros)
                    curr_obs_feature = np.array(data_pt[1:obs_feature_size+1]).reshape((obs_hist_size, obs_unit_feature_size))
                    curr_obs_feature = np.flip(curr_obs_feature, 0)
                    curr_obs_pos = np.zeros((1, obs_hist_size, 2))
                    # (1 x max_obs_hist_size x 2)
                    curr_obs_pos[0, -curr_obs_hist_size:, :] = curr_obs_feature[-curr_obs_hist_size:, 1:3]
                    self.obs_pos.append(curr_obs_pos)
                    curr_obs_pos_rel =  np.zeros((1, obs_hist_size, 2))
                    curr_obs_pos_rel[0, -curr_obs_hist_size+1:, :] = \
                        curr_obs_pos[0, -curr_obs_hist_size+1:, :] - curr_obs_pos[0, -curr_obs_hist_size:-1, :]
                    self.obs_pos_rel.append(curr_obs_pos_rel)

                    # Get the obstacle polygon features (organized from past to present).
                    curr_obs_polygon = curr_obs_feature[:, -40:].reshape((1, obs_hist_size, 20, 2))
                    self.obs_polygons.append(curr_obs_polygon)

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

                    # Skip getting label data for those without labels at all.
                    if len(data_pt) <= obs_feature_size+1+(single_lane_feature_size)*curr_num_lane_sequence:
                        self.is_predictable.append(np.zeros((1, 1)))
                        self.reference_world_coord.append([0.0, 0.0, 0.0])
                        self.future_traj.append(np.zeros((1, 30, 2)))
                        self.future_traj_rel.append(np.zeros((1, 29, 2)))
                        continue
                    self.is_predictable.append(np.ones((1, 1)))

                    # Get the future trajectory label.
                    curr_future_traj = np.array(data_pt[-91:-31]).reshape((2, 30))
                    curr_future_traj = curr_future_traj.transpose()
                    ref_world_coord = [curr_future_traj[0, 0], curr_future_traj[0, 1], data_pt[-31]]
                    self.reference_world_coord.append(ref_world_coord)
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

                self.end_idx.append(accumulated_data_pt)
                if self.end_idx[-1] == self.start_idx[-1]:
                    self.end_idx.pop()
                    self.start_idx.pop()

        self.total_num_data_pt = len(self.start_idx)
        print ('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        if self.img_mode:
            s_idx = self.start_idx[idx]
            e_idx = self.end_idx[idx]
            obs_hist_sizes = np.concatenate(self.obs_hist_sizes[s_idx:e_idx])
            obs_polygons = np.concatenate(self.obs_polygons[s_idx:e_idx])

            predictable_prob = np.concatenate(self.is_predictable[s_idx:e_idx])
            predictable_prob = predictable_prob.reshape((-1))
            predictable_prob = predictable_prob / np.sum(predictable_prob)
            predicting_idx = np.random.choice(predictable_prob.shape[0], 1, p=predictable_prob)[0]
            world_coord = self.reference_world_coord[s_idx + predicting_idx]
            obs_future_traj = self.future_traj[s_idx + predicting_idx][0,:,:]
            region = self.map_region[s_idx + predicting_idx]
            # TODO(Hongyi): modify the following part to include multiple obstacles.
            obs_mapping = ObstacleMapping(region, self.base_map[region], world_coord, obs_polygons)
            img = obs_mapping.crop_by_history(obs_polygons[predicting_idx])
            # cv.imwrite("./test/img{}.png".format(idx), img)
            if self.img_transform:
                img = self.img_transform(img)
            obs_positions = np.concatenate(self.obs_pos[s_idx:e_idx])
            obs_pos = obs_positions[predicting_idx, :, :]
            return (img, torch.from_numpy(obs_pos).float()), torch.from_numpy(obs_future_traj).float()
        else:
            s_idx = self.start_idx[idx]
            e_idx = self.end_idx[idx]
            out = (np.concatenate(self.obs_hist_sizes[s_idx:e_idx]), \
                   np.concatenate(self.obs_pos[s_idx:e_idx]), \
                   np.concatenate(self.obs_pos_rel[s_idx:e_idx]), \
                   np.concatenate(self.lane_feature[s_idx:e_idx]), \
                   np.concatenate(self.future_traj[s_idx:e_idx]), \
                   np.concatenate(self.future_traj_rel[s_idx:e_idx]), \
                   np.concatenate(self.is_predictable[s_idx:e_idx]),
                   (e_idx-s_idx)*np.ones((1,1)))
            return out

def collate_fn(batch):
    '''
    return:
        - obs_hist_size: N x 1
        - obs_pos: N x max_obs_hist_size x 2
        - obs_pos_rel: N x max_obs_hist_size x 2
        - lane_features: M x 150 x 4
        - same_obstacle_mask: M x 1

        - future_traj: N x 30 x 2
        - future_traj_rel: N x 29 x 2
        - is_predictable: N x 1
    '''
    # batch is a list of tuples.
    # unzip to form lists of np-arrays.
    obs_hist_size, obs_pos, obs_pos_rel, lane_features, future_traj, future_traj_rel, is_predictable, same_scene_mask = zip(*batch)

    same_obstacle_mask = [elem.shape[0] for elem in lane_features]
    obs_hist_size = np.concatenate(obs_hist_size)
    obs_pos = np.concatenate(obs_pos)
    obs_pos_rel = np.concatenate(obs_pos_rel)
    lane_features = np.concatenate(lane_features)
    future_traj = np.concatenate(future_traj)
    future_traj_rel = np.concatenate(future_traj_rel)

    same_obstacle_mask = [np.ones((length, 1))*i for i, length in enumerate(same_obstacle_mask)]
    same_obstacle_mask = np.concatenate(same_obstacle_mask)

    #TODO(jiacheng): process the same_scene_mask.

    return (torch.from_numpy(obs_hist_size), torch.from_numpy(obs_pos), torch.from_numpy(obs_pos_rel), \
            torch.from_numpy(lane_features).float(), torch.from_numpy(same_obstacle_mask)), \
           (torch.from_numpy(future_traj), torch.from_numpy(future_traj_rel), torch.ones(obs_pos.shape[0], 1))


if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.
    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data('/home/jiacheng/large-data/data_preprocessing/features/')
