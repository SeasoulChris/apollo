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
from learning_algorithms.utilities.network_utils import *


class PointToLineProjection(nn.Module):
    '''Get the projection point from a given point to a given line-segment.
    '''
    def __init__(self):
        super(PointToLineProjection, self).__init__()

    def forward(self, line_seg_start, line_seg_end, point):
        '''
        params:
            - line_seg_start: (N x 2)
            - line_seg_end: (N x 2)
            - point: (N x 2)

        return:
            - projected_point: (N x 2)
            - dist: (N x 1)
        '''
        N = point.size(0)
        # (N x 2)
        line_vec = line_seg_end - line_seg_start
        # (N x 1)
        line_vec_mag = torch.sqrt(torch.sum(line_vec**2, 1)).view(N, 1)
        # (N x 2)
        unit_line_vec = line_vec / line_vec_mag.repeat(1, 2)

        # (N x 2)
        point_vec = point - line_seg_start
        # (N x 1)
        projected_point_vec_mag = torch.sum(point_vec * unit_line_vec, 1).view(N, 1)
        # (N x 2)
        projected_point_vec = projected_point_vec_mag.repeat(1, 2) * unit_line_vec
        # (N x 2)
        projected_point = line_seg_start + projected_point_vec
        # (N x 1)
        dist = torch.sqrt(torch.sum((point - projected_point)**2, 1)).view(N, 1)

        return projected_point, dist


class FindClosestNodeFromLineToPoint(nn.Module):
    '''Given a line(consisting of multiple nodes), find the node with the
       shortest distance to a given point.
    '''
    def __init__(self):
        super(FindClosestNodeFromLineToPoint, self).__init__()

    def forward(self, line_nodes, point):
        '''
        params:
            - line_nodes: N x num_node x 2
            - point: N x 2

        return:
            - idx of the min dist node: N
        '''
        N = point.size(0)
        num_node = line_nodes.size(1)
        # (N x num_node-2 x 2)
        nodes = line_nodes[:, 1:-1, :2].float()

        # Calculate the L2 distance between every lane-points to the point.
        # (N x num_node-2 x 2)
        distances = nodes - point.view(N,1,2).repeat(1,num_node-2,1).float()
        distances = distances ** 2
        # (N x num_node-2)
        distances = torch.sum(distances, 2)

        # Figure out the idx of the lane-point that's closest to obstacle.
        # (N)
        min_idx = torch.argmin(distances, dim=1)

        return min_idx + 1


class FindClosestLineSegmentFromLineToPoint(nn.Module):
    def __init__(self):
        super(FindClosestLineSegmentFromLineToPoint, self).__init__()
        self.find_closest_idx = FindClosestNodeFromLineToPoint()

    def forward(self, line_nodes, point):
        '''
        params:
            - line_nodes: N x num_node x 2
            - point: N x 2

        return:
            - start idx of the min dist node: N
            - end idx of the min dist node: N
        '''
        N = line_nodes.size(0)
        # (N)
        min_idx = self.find_closest_idx(line_nodes, point)
        # (N x 2)
        min_node = line_nodes[torch.arange(N), min_idx, :]

        # (N)
        min_idx_prev = min_idx - 1
        # (N x 2)
        min_node_prev = line_nodes[torch.arange(N), min_idx_prev, :]
        # (N)
        dist_to_prev = torch.sum((min_node_prev - min_node) ** 2, 1)

        # (N)
        min_idx_next = min_idx + 1
        # (N x 2)
        min_node_next = line_nodes[torch.arange(N), min_idx_next, :]
        # (N )
        dist_to_next = torch.sum((min_node_next - min_node) ** 2, 1)

        # Get the 2nd minimum distance node's index.
        min_idx_2nd = min_idx_prev
        idx_to_be_modified = (dist_to_next < dist_to_prev)
        if torch.sum(idx_to_be_modified).long().item() != 0:
            min_idx_2nd[idx_to_be_modified] = min_idx_next[idx_to_be_modified]

        # Return the correct order
        min_indices = torch.cat((min_idx.view(N,1), min_idx_2nd.view(N,1)), 1)
        idx_before, _ = torch.min(min_indices, dim=1)
        idx_after, _ = torch.max(min_indices, dim=1)
        return idx_before, idx_after
