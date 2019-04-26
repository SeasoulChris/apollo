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

from learning_algorithms.utilities.train_utils import *

class SocialInteraction(nn.Module):
    def __init__(self):
        # Spatial processing
        # Temporal processing
        return

    def forward(self, X):
        # 1. Update variables based on current time-stamp info.

        # 2. Look into the scope and gather needed info.

        # 3. Aggregate and update each node based on #2

        # 4. If finished gathering info, predict for the next time-stamp.

        return X


class SocialPooling(nn.Module):
    def __init__(self, grid_size=2, area_span=1.6):
        super(SocialPooling, self).__init__()
        self.grid_size = grid_size
        self.area_span = area_span

    def decide_grid(self, curr_pos_t):
        '''Helper function that decides the grid for social-pooling.

            curr_pos_t: (N x 1 x 2)
        '''
        N = curr_pos_t.size(0)
        eps = 1e-2

        # For this matrix (N x N x 2), the (i,j) element indicates the relative
        # position of agent j w.r.t. agent i.
        rel_pos_matrix = torch.transpose(curr_pos_t.repeat(1, N ,1), 0, 1) -\
                         curr_pos_t.repeat(1, N, 1)

        # (N x N) matrix of mask, the (i,j) element indicates whether agent j
        # is within pooling area of agent i. (note that agent i is not within
        # pooling area of agent i, which is itself)
        mask_within_pooling_area = \
            (rel_pos_matrix[:, :, 0] < self.area_span / 2.0-eps) * \
            (rel_pos_matrix[:, :, 0] > -self.area_span / 2.0+eps) * \
            (rel_pos_matrix[:, :, 1] < self.area_span / 2.0-eps) * \
            (rel_pos_matrix[:, :, 1] > -self.area_span / 2.0+eps)
        mask_within_pooling_area = mask_within_pooling_area.float()
        mask_within_pooling_area -= cuda(torch.eye(N))

        # (N x N) matrix of mask, the (i,j) element indicates which grid that
        # agent j falls in w.r.t. agent i.
        mask_grid_id = torch.floor(
            (rel_pos_matrix.float() + torch.tensor(self.area_span / 2.0)) / \
            torch.tensor(self.area_span / self.grid_size))
        mask_grid_id = mask_grid_id[:, :, 0] * self.grid_size + mask_grid_id[:, :, 1]
        mask_grid_id *= mask_within_pooling_area
        mask_grid_id = mask_grid_id.long()

        return mask_within_pooling_area, mask_grid_id

    def forward(self, ht, pos_t, same_scene_mask):
        '''
            ht: the matrix of hidden states (N x 1 x hidden_size)
            pos_t: the matrix of current positions (N x 1 x 2)
            same_scene_mask: the mask indicating what agents are in the same scene.
                agents that are in the same scene share the same number.
                (N x 1) --> e.g. [1,1,1,2,2,2,2,2,2,3,3, ...]
        '''
        N = ht.size(0)
        hidden_size = ht.size(2)

        ht_pooled = cuda(torch.zeros(N, self.grid_size ** 2, hidden_size))
        all_scene_ids = torch.unique(same_scene_mask).cpu().numpy().tolist()
        N_filled = 0

        # Go through every scene one by one.
        for scene_id in all_scene_ids:
            # 1. get the ht and pos_t for this scene_id:
            # (curr_N x 1 x hidden_size)
            curr_ht = ht[same_scene_mask[:,0] == scene_id, :, :]
            # (curr_N x 1 x 2)
            curr_pos_t = pos_t[same_scene_mask[:,0] == scene_id, :, :]
            curr_N = curr_ht.size(0)
            if (curr_N == 0):
                continue
            curr_ht = curr_ht.view(curr_N, 1, -1)
            curr_pos_t = curr_post_t.view(curr_N, 1, 2)

            # 2. get the pooling grid matrix.
            mask_within_pooling_area, mask_grid_id = self.decide_grid(curr_pos_t)
            # ht_matrix is of size (curr_N x curr_N x hidden_size)
            ht_matrix = torch.transpose(curr_ht.repeat(1, curr_N, 1), 0, 1).float()

            # 3. Only retain those that are within pooling area.
            ht_matrix *= mask_within_pooling_area.\
                reshape((curr_N, curr_N, 1)).repeat(1, 1, hidden_size)

            # 4. Use scatter_add to add up those that are in the same grid.
            # (curr_N x grid_size^2 x hidden_size)
            mask_grid_id = mask_grid_id.\
                reshape((curr_N, curr_N, 1)).repeat(1, 1, hidden_size)
            # (curr_N x grid_size^2 x hidden_size)
            curr_ht_pooled = cuda(torch.zeros(curr_N, self.grid_size ** 2, hidden_size))
            curr_ht_pooled = curr_ht_pooled.scatter_add(1, mask_grid_id, ht_matrix)

            # 5. Update the pooled Ht.
            ht_pooled[N_filled:N_filled+curr_N, :, :] = curr_ht_pooled
            N_filled += curr_N
        
        return ht_pooled
