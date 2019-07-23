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
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from torchvision import models
from torchvision import transforms

sys.path.append('../../utilities')

from helper_utils import *

'''
========================================================================
Dataset set-up
========================================================================
'''

class JunctionMapDataset(Dataset):
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


        try:
            key = os.path.basename(img_name).replace(".png","")
            # pos_dict = np.load(os.path.join(os.path.dirname(img_name),'obs_pos.npy')).item()
            # past_pos = pos_dict[key]
            # label_dict = np.load(os.path.join(os.path.dirname(img_name).replace("image-feature","features-san-mateo-new"),'future_status.npy')).item()
            # future_pos = label_dict[key]
            # origin = future_pos[0]
            junction_label_dict = np.load(os.path.join(os.path.dirname(img_name).replace("image-feature","labels-san-mateo"),'junction_label.npy')).item()
            # past_pos = [world_coord_to_relative_coord(pos, origin) for pos in past_pos]
            # future_pos = [world_coord_to_relative_coord(pos, origin) for pos in future_pos]
            # sample_obs_feature = torch.FloatTensor(past_pos).view(-1)
            junction_label = junction_label_dict[key]
            if len(junction_label)!=24:
                return self.__getitem__(idx+1)
            sample_mask = torch.FloatTensor(junction_label[-12:]).view(-1)
            sample_label = torch.FloatTensor(junction_label[:12]).view(-1)
        except:
            return self.__getitem__(idx+1)

        if len(sample_mask) != 12 or len(sample_label) != 12:
            return self.__getitem__(idx+1)

        return (sample_img, sample_mask), sample_label


'''
========================================================================
Model definition
========================================================================
'''
class JunctionMapModel(nn.Module):
    def __init__(self, obs_feature_size, output_dim,
                 cnn_net=models.resnet50, pretrained=True):
        super(JunctionMapModel, self).__init__()

        self.cnn = cnn_net(pretrained=pretrained)
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, 60),
            nn.Dropout(0.3),
            nn.Linear(60, output_dim),
            # nn.Softmax()
        )
    
    def forward(self, X):
        img, obs_mask = X
        out = self.cnn(img)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = torch.mul(out, obs_mask)
        out = F.softmax(out)
        return out

class JunctionMapLoss():
    def loss_fn(self, y_pred, y_true):
        true_label = y_true.topk(1)[1].view(-1)
        loss_func = nn.CrossEntropyLoss()
        return loss_func(y_pred, true_label)

    def loss_info(self, y_pred, y_true):
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        pred_label = y_pred.topk(1)[1]
        true_label = y_true.topk(1)[1]
        accuracy = (pred_label == true_label).type(torch.float).mean().item()
        print("Accuracy is {:.3f} %".format(100*accuracy))
        return