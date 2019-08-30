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

import cv2 as cv
import glob
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from torchvision import models
from torchvision import transforms

from learning_algorithms.utilities.helper_utils import *
from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *

'''
========================================================================
Dataset set-up
========================================================================
'''


class SemanticMapDataset(Dataset):
    def __init__(self, dir, transform=None, verbose=False):
        self.items = glob.glob(dir+"/**/*.png", recursive=True)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        self.verbose = verbose

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name = self.items[idx]
        sample_img = cv.imread(img_name)
        if sample_img is None:
            print('Failed to load' + self.items[idx])
        if self.transform:
            sample_img = self.transform(sample_img)

        # TODO(jiacheng): implement this.
        try:
            key = os.path.basename(img_name).replace(".png", "")
            pos_dict = np.load(os.path.join(os.path.dirname(img_name), 'obs_pos.npy')).item()
            past_pos = pos_dict[key]
            label_dict = np.load(os.path.join(os.path.dirname(img_name).replace(
                "image-feature", "features-san-mateo-new").replace("image-valid", "features-san-mateo-new"), 'future_status.npy')).item()
            future_pos = label_dict[key]
            origin = future_pos[0]
            past_pos = [world_coord_to_relative_coord(pos, origin) for pos in past_pos]
            future_pos = [world_coord_to_relative_coord(pos, origin) for pos in future_pos]

            sample_obs_feature = torch.FloatTensor(past_pos).view(-1)
            sample_label = torch.FloatTensor(future_pos[0:10]).view(-1)
        except:
            return self.__getitem__(idx+1)

        if len(sample_obs_feature) != 20 or len(sample_label) != 20:
            return self.__getitem__(idx+1)

        return (sample_img, sample_obs_feature), sample_label


'''
========================================================================
Model definition
========================================================================
'''


class SemanticMapModel(nn.Module):
    def __init__(self, num_pred_points, num_history_points,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapModel, self).__init__()

        self.cnn = cnn_net(pretrained=pretrained)
        self.num_pred_points = num_pred_points
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features + num_history_points * 2, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, num_pred_points * 2)
        )

    def forward(self, X):
        img, obs_pos, _, _ = X
        out = self.cnn(img)
        out = out.view(out.size(0), -1)
        obs_pos = obs_pos.view(obs_pos.size(0), -1)
        out = torch.cat([out, obs_pos], -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.num_pred_points, 2)
        return out


class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        out = y_pred - y_true
        out = torch.mean(out ** 2)
        return out


class SemanticMapSelfLSTMModel(nn.Module):
    def __init__(self, pred_len, num_history_points,
                 embed_size=64, hidden_size=128,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapSelfLSTMModel, self).__init__()
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = self.cnn.fc.in_features
        self.pred_len = pred_len
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.disp_embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=1, batch_first=True)

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 2),
        )

    def forward(self, X):
        img, obs_pos, obs_hist_size, obs_pos_rel = X
        N = obs_pos.size(0)
        observation_len = obs_pos.size(1)
        ht, ct = self.h0.repeat(N, 1), self.h0.repeat(N, 1)

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 2))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))

        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                ts_obs_mask = (obs_hist_size > observation_len-t).long().view(-1)
                curr_obs_pos_rel = obs_pos_rel[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                ts_obs_mask = pred_mask
                pred_input = torch.cat((ht.clone(), img_embedding), 1)
                pred_out[:, t-observation_len, :] = self.pred_layer(pred_input).float().clone()
                curr_obs_pos_rel = pred_out[:, t-observation_len, :2]
                curr_obs_pos = curr_obs_pos + curr_obs_pos_rel
                pred_traj[:, t-observation_len, :] = curr_obs_pos.clone()

            curr_N = torch.sum(ts_obs_mask).long().item()
            if curr_N == 0:
                continue

            ts_obs_mask = (ts_obs_mask == 1)
            disp_embedding = self.disp_embed(
                (curr_obs_pos_rel[ts_obs_mask, :]).clone()).view(curr_N, 1, -1)

            _, (ht_new, ct_new) = self.lstm(
                disp_embedding, (ht[ts_obs_mask, :].view(1, curr_N, -1), ct[ts_obs_mask, :].view(1, curr_N, -1)))
            ht[ts_obs_mask, :] = ht_new.view(curr_N, -1)
            ct[ts_obs_mask, :] = ct_new.view(curr_N, -1)

        return pred_traj
