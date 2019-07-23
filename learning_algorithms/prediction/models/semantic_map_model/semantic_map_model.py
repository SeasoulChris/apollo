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
            key = os.path.basename(img_name).replace(".png","")
            pos_dict = np.load(os.path.join(os.path.dirname(img_name),'obs_pos.npy')).item()
            past_pos = pos_dict[key]
            label_dict = np.load(os.path.join(os.path.dirname(img_name).replace("image-feature","features-san-mateo-new").replace("image-valid","features-san-mateo-new"),'future_status.npy')).item()
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
    def __init__(self, num_pred_points, obs_feature_size,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapModel, self).__init__()

        self.cnn = cnn_net(pretrained=pretrained)
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features + obs_feature_size, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 130),
            nn.Dropout(0.3),
            nn.Linear(130, num_pred_points * 2)
        )
    
    def forward(self, X):
        img, obs_feature = X
        out = self.cnn(img)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, obs_feature], 1)
        return self.fc(out)

class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        out = y_pred - y_true
        out = torch.mean(out ** 2)
        return out
