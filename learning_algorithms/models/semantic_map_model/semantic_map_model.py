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
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

from torchvision import models
from torchvision import transforms

from utilities.IO_utils import *


'''
========================================================================
Dataset set-up
========================================================================
'''
def process_img(filepath, transform, verbose=False):
    img = torch.from_numpy(cv.imread(filepath))
    img = img.permute((2, 0, 1))

    norm_func = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.float() / 255.0
    img = norm_func(img)

    if transform is not None:
        img = transform(img)
    
    return img

class SemanticMapDataset(Dataset):
    def __init__(self, dir, transform=None, is_simple_dataloader=False,
                 verbose=False):
        self.all_files = GetListOfFiles(dir)
        self.transform = transform
        self.is_simple_dataloader = is_simple_dataloader
        self.verbose = verbose

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        sample_img = process_img(self.all_files[idx], self.transform, self.verbose)
        if sample_img is None:
            print('Failed to load' + self.items[idx])

        # TODO(jiacheng): implement this.
        sample_obs_feature = None

        return sample_img, sample_obs_feature


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
