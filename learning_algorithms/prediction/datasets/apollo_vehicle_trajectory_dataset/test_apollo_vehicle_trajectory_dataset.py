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

import argparse
import cv2 as cv
import logging
import numpy as np
import os
import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

import learning_algorithms.prediction.datasets.apollo_vehicle_trajectory_dataset.apollo_vehicle_trajectory_dataset as apollo_vehicle_trajectory_dataset
from apollo_vehicle_trajectory_dataset import ApolloVehicleTrajectoryDataset as ApolloVehicleTrajectoryDataset
from apollo_vehicle_trajectory_dataset import collate_fn as collate_fn
from learning_algorithms.prediction.models.lane_attention_trajectory_model.lane_attention_trajectory_model import *
from learning_algorithms.prediction.models.semantic_map_model.semantic_map_model import *


def point_to_idx(point_x, point_y):
        return (int((point_x + 40)/0.1), int((point_y + 40)/0.1))

def plot_img(obs_features, lane_features, labels, pred, count):
    # black background
    img = np.zeros([1000, 800, 3], dtype=np.uint8)

    # draw boundaries
    cv.circle(img,(400,400),2,color=[255,255,255], thickness=4)
    cv.line(img, (0,0), (799,0), color=[255, 255, 255])
    cv.line(img, (799,0), (799,999), color=[255, 255, 255])
    cv.line(img, (0,999), (0,0), color=[255, 255, 255])
    cv.line(img, (799,999), (0,999), color=[255, 255, 255])
    num_lane_seq = lane_features.shape[0]

    for obs_hist_pt in range(20):
        cv.circle(img, point_to_idx(obs_features[1+obs_hist_pt*2], obs_features[obs_hist_pt*2]), radius=3, color=[128,128,128])

    for lane_idx in range(num_lane_seq):
        curr_lane = lane_features[lane_idx]
        color_to_use = [0, 128, 128]
        for point_idx in range(149):
            cv.line(img, point_to_idx(curr_lane[point_idx,1], curr_lane[point_idx,0]), \
                point_to_idx(curr_lane[point_idx+1,1], curr_lane[point_idx+1,0]), \
                color=color_to_use)

    for obs_future_pt in range(30):
        cv.circle(img, point_to_idx(labels[1+obs_future_pt*2], labels[obs_future_pt*2]), radius=3, color=[128,128,0])
        cv.circle(img, point_to_idx(pred[1+obs_future_pt*2], pred[obs_future_pt*2]), radius=3, color=[0,128,128])

    cv.imwrite('img={}__laneseq={}.png'.format(count, num_lane_seq), cv.flip(cv.flip(img,0),1))


if __name__ == '__main__':
    # Unit test without image
    dataset_path = '/home/jiacheng/large-data/data_preprocessing/training_data/features/san_mateo/2018/2018-10-01'
    test_dataset = ApolloVehicleTrajectoryDataset(dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    count = 0
    for i, (X, y) in enumerate(test_dataloader):
        if count == 10:
            break
        # print ('=======================')
        # print (y[0])
        # print (y[1])
        # print (y[2])
        model = SemanticMapSelfLSTMModel(30, 20)
        model_path = "/home/sunhongyi/Documents/apollo-fuel/learning_algorithms/prediction/pipelines/vehicle_trajectory_prediction_pipeline/model_epoch1_valloss8798324149956.923828.pt"
        model.load_state_dict(torch.load(model_path))
        model.cuda().eval()
        pred = model.forward(cuda(X))
        obs_features = X[1].detach().numpy().reshape(-1)
        lane_features = X[1].detach().numpy().reshape(-1)
        labels = y.numpy().reshape(-1)
        pred = pred.cpu().detach().numpy().reshape(-1)
        plot_img(obs_features, lane_features, labels, pred, count)
        count += 1

    # Unit test with image
    dataset_path = '/data/san_mateo/'
    test_dataset = ApolloVehicleTrajectoryDataset(dataset_path, True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    count = 0
    for i, ((img,_,_,_), traj) in enumerate(test_dataloader):
        # if count == 10:
        #     break
        print(traj)
        # cv.imwrite("test/test{}.png".format(i),np.array(img.view(224,224,3)))
        count += 1
        print(count, img.size())
