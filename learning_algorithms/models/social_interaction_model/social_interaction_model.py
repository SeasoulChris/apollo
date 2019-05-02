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
    def __init__(self, pred_len=12, grid_size=2, area_span=2.0):
        # Spatial processing
        # Temporal processing
        return

    def forward(self, X):
        # 1. Update variables based on current time-stamp info.

        # 2. Look into the scope and gather needed info.

        # 3. Aggregate and update each node based on #2

        # 4. If finished gathering info, predict for the next time-stamp.

        return X


class SimpleLSTM(nn.Module):
    def __init__(self, pred_len=12, embed_size=64, hidden_size=128):
        super(SimpleLSTM, self).__init__()
        self.pred_len = pred_len
        self.pos_embedding = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size, 5),
        )

    def forward(self, X):
        return X



class SocialLSTM(nn.Module):
    def __init__(self, pred_len=12, grid_size=2, area_span=2.0,
                 embed_size=64, hidden_size=128):
        super(SocialLSTM, self).__init__()

        self.pred_len = pred_len

        self.pos_embedding = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.social_embedding = torch.nn.Sequential(
            nn.Linear(grid_size * grid_size * hidden_size, embed_size),
            nn.ReLU(),
        )

        self.social_pooling = SocialPooling(grid_size, area_span)

        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers=1, batch_first=True)
        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size, 5),
        )

    def forward(self, X):
        # Get dimensions
        past_traj, past_traj_rel, past_traj_timestamp_mask, is_predictable, same_scene_mask = X
        N = past_traj.size(0)
        observation_len = past_traj.size(1)

        # Look at the past trajectory
        # (N x 1 x self.hidden_size)
        ht, ct = self.h0.repeat(N, 1, 1), self.c0.repeat(N, 1, 1)
        for t in range(observation_len):
            # Get the related variables at this timestamp.
            curr_mask = (past_traj_timestamp_mask[:, t] == 1)
            curr_N = torch.sum(curr_mask).item()
            # (curr_N x 1 x 2)
            curr_point = past_traj[curr_mask, t, :].reshape(curr_N, 1, 2)
            curr_point_rel = past_traj_rel[curr_mask, t, :].reshape(curr_N, 2)
            # (curr_N x 1)
            curr_same_scene_mask = same_scene_mask[curr_mask]
            # (1 x curr_N x hidden_size)
            curr_ht, curr_ct = ht[curr_mask, :, :], ct[curr_mask, :, :]

            # Apply social-pooling
            # (curr_N x grid_size^2 x hidden_size)
            Ht = self.social_pooling(curr_ht, curr_point, curr_same_scene_mask)

            # Apply embeddings
            # (curr_N x embed_size)
            et = self.pos_embedding(curr_point_rel.float())
            at = self.social_embedding(Ht.view(curr_N, -1))

            # Step through RNN
            _, (curr_ht, curr_ct) = self.lstm(
                torch.cat((et, at), 1).view(curr_N, 1, -1), 
                (curr_ht.view(1, curr_N, -1), curr_ct.view(1, curr_N, -1)))
            ht[curr_mask, :, :], ct[curr_mask, :, :] = \
                curr_ht.view(curr_N, 1, -1), curr_ct.view(curr_N, 1, -1)

        # Predict the future trajectory
        pred_mask = (past_traj_timestamp_mask[:, -1] == 1)
        pred_N = torch.sum(pred_mask).item()
        if pred_N == 0:
            return None
        pred_same_scene_mask = same_scene_mask[pred_mask]
        pred_point = past_traj[pred_mask, -1, :].float().reshape(pred_N, 1, 2)
        pred_ht, pred_ct = ht[pred_mask, :, :], ct[pred_mask, :, :]
        # (pred_N x pred_len x (ux, uy, sigma_x, sigma_y, rho))
        pred_out = cuda(torch.zeros(pred_N, self.pred_len, 5))
        # (pred_N x pred_len x 2)
        pred_traj = cuda(torch.zeros(pred_N, self.pred_len, 2))
        for t in range(self.pred_len):
            pred_out[:, t, :] = self.pred_layer(pred_ht.view(pred_N, -1)).view(pred_N, 5).float()
            pred_point_rel = cuda(pred_out[:, t, :2].float().view(pred_N, 1, 2))
            pred_point = pred_point + pred_point_rel
            pred_traj[:, t, :] = pred_point.reshape(pred_N, 2)
            pred_point_rel = pred_point_rel.view(pred_N, 2).clone()

            Ht = self.social_pooling(pred_ht, pred_point, pred_same_scene_mask)
            et = self.pos_embedding(pred_point_rel)
            at = self.social_embedding(Ht.view(pred_N, -1))
            _, (pred_ht, pred_ct) = self.lstm(
                torch.cat((et, at), 1).view(pred_N, 1, -1),
                (pred_ht.view(1, pred_N, -1), pred_ct.view(1, pred_N, -1)))
            pred_ht = pred_ht.view(pred_N, 1, -1)
        pred_out_all = cuda(torch.zeros(N, self.pred_len, 5))
        pred_out_all[pred_mask, :, :] = pred_out
        pred_traj_all = cuda(torch.zeros(N, self.pred_len, 2))
        pred_traj_all[pred_mask, :, :] = pred_traj
        return pred_out_all[is_predictable[:, 0] == 1, :, :],\
               pred_traj_all[is_predictable[:, 0] == 1, :, :]


class SocialPooling(nn.Module):
    '''The social-pooling module used in the paper of Social-LSTM.
    '''
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
                (N x 1) --> e.g. [0,0,0,0,0,0,1,1,1,2,2,2,2,2,2,3,3, ...]
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
            curr_pos_t = curr_pos_t.view(curr_N, 1, 2)

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


class SocialAttention(nn.Module):
    '''The social-attention model
    '''
    def __init__(self, embed_size=64, edge_hidden_size=256, node_hidden_size=128, pred_len=12):
        super(SocialAttention, self).__init__()

        self.pred_len = pred_len

        # Initialize initial states for spatial-edge RNN.
        s_edge_h0 = torch.zeros(1, 1, edge_hidden_size)
        s_edge_c0 = torch.zeros(1, 1, edge_hidden_size)
        nn.init.xavier_normal_(s_edge_h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(s_edge_c0, gain=nn.init.calculate_gain('relu'))
        self.s_edge_h0 = nn.Parameter(s_edge_h0, requires_grad=True)
        self.s_edge_c0 = nn.Parameter(s_edge_c0, requires_grad=True)

        # Initialize initial states for temporal-edge RNN.
        t_edge_h0 = torch.zeros(1, edge_hidden_size)
        t_edge_c0 = torch.zeros(1, edge_hidden_size)
        nn.init.xavier_normal_(t_edge_h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(t_edge_c0, gain=nn.init.calculate_gain('relu'))
        self.t_edge_h0 = nn.Parameter(t_edge_h0, requires_grad=True)
        self.t_edge_c0 = nn.Parameter(t_edge_c0, requires_grad=True)

        # Initialize initial states for node RNN.
        node_h0 = torch.zeros(1, node_hidden_size)
        node_c0 = torch.zeros(1, node_hidden_size)
        nn.init.xavier_normal_(node_h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(node_c0, gain=nn.init.calculate_gain('relu'))
        self.node_h0 = nn.Parameter(node_h0, requires_grad=True)
        self.node_c0 = nn.Parameter(node_c0, requires_grad=True)

        #
        self.spatial_edge_rnn = SpatialEdgeRNN(embed_size, edge_hidden_size)
        self.temporal_edge_rnn = TemporalEdgeRNN(embed_size, edge_hidden_size)
        self.pred_layer = torch.nn.Sequential(
            nn.Linear(edge_hidden_size, 5),
        )

    def forward(self, X):
        past_traj, past_traj_rel, past_traj_timestamp_mask, is_predictable, same_scene_mask = X
        N = past_traj.size(0)
        observation_len = past_traj.size(1)

        # INITIALIZATION:
        # Create a list of square matrices to hold hidden-states for spatial edges. (h_{uv})
        # e.g. if same_scene_mask is [0,0,1,1,1,2], then the resulting ct/ht_list
        #      is a list of matrices [2x2, 3x3, 1x1].
        s_edge_ht_list = []
        s_edge_ct_list = []
        for i in range(same_scene_mask.max().long().item()+1):
            curr_dim = torch.sum(same_scene_mask==i).item()
            s_edge_ht_list.append(self.s_edge_h0.repeat(curr_dim, curr_dim, 1))
            s_edge_ct_list.append(self.s_edge_c0.repeat(curr_dim, curr_dim, 1))
        # Create a vector of hidden-states for temporal edges. (h_{vv})
        # (N x edge_hidden_size)
        t_edge_ht_list = self.t_edge_h0.repeat(N, 1)
        t_edge_ct_list = self.t_edge_c0.repeat(N, 1)
        # Create a vector of hidden-states for nodes (h_v)
        # (N x edge_hidden_size)
        node_ht_list = self.node_h0.repeat(N, 1)
        node_ct_list = self.node_c0.repeat(N, 1)

        # RUNNING THROUGH EACH TIME-STAMP:
        pred_mask = (past_traj_timestamp_mask[:, -1] == 1)
        pred_N = torch.sum(pred_mask).item()
        if pred_N == 0:
            return None
        pred_out = cuda(torch.zeros(pred_N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(pred_N, self.pred_len, 2))
        curr_node_mask, curr_node_N, curr_point, curr_point_rel = None, None, None, None
        for t in range(observation_len+self.pred_len):
            if t < observation_len:
                curr_node_mask = (past_traj_timestamp_mask[:, t] == 1)
                curr_node_N = torch.sum(curr_node_mask).item()
                # (curr_node_N x 2)
                curr_point = past_traj[curr_node_mask, t, :].reshape(curr_node_N, 2).float()
                curr_point_rel = past_traj_rel[curr_node_mask, t, :].reshape(curr_node_N, 2).float()
            else:
                curr_node_mask = pred_mask
                curr_node_N = pred_N
                pred_out[:, t-observation_len, :] = self.pred_layer(
                    t_edge_ht_list[pred_mask]).float()
                curr_point_rel = pred_out[:, t-observation_len, :2].float().view(pred_N, 2).float().clone()
                curr_point = curr_point + curr_point_rel
                pred_traj[:, t-observation_len, :] = curr_point

            # 1. Do spatial-edge (h_{uv}) RNN.
            if t < observation_len:
                s_edge_ht_list, s_edge_ct_list = self.spatial_edge_rnn(\
                    s_edge_ht_list, s_edge_ct_list, past_traj[:, t, :],\
                    past_traj_timestamp_mask[:, t].view(N, 1),\
                    same_scene_mask)

            # 2. Do temporal-edge (h_{vv}) RNN.
            # (curr_node_N x edge_hidden_size)
            curr_t_edge_ht_list = t_edge_ht_list[curr_node_mask, :]
            curr_t_edge_ct_list = t_edge_ct_list[curr_node_mask, :]
            curr_t_edge_ht_list, curr_t_edge_ct_list =\
                self.temporal_edge_rnn(curr_t_edge_ht_list, curr_t_edge_ct_list, curr_point_rel)
            t_edge_ht_list[curr_node_mask, :] = curr_t_edge_ht_list
            t_edge_ct_list[curr_node_mask, :] = curr_t_edge_ct_list

            # 3. Do EdgeToNodeAttention.
            # TODO:(jiacheng) to be implemented.

            # 4. Aggregate and update nodes (h_v).
            # TODO:(jiacheng) to be implemented.

        pred_out_all = cuda(torch.zeros(N, self.pred_len, 5))
        pred_out_all[pred_mask, :, :] = pred_out
        pred_traj_all = cuda(torch.zeros(N, self.pred_len, 2))
        pred_traj_all[pred_mask, :, :] = pred_traj
        return pred_out_all[is_predictable[:, 0] == 1, :, :],\
               pred_traj_all[is_predictable[:, 0] == 1, :, :]


class SpatialEdgeRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=256):
        super(SpatialEdgeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, ht_list, ct_list, traj, timestamp_mask, same_scene_mask):
        '''The forward function for spatial-edge RNN.

            ht_list: list of matrices [2x2, 4x4, 1x1] x hidden_size
            ct_list: lsit of matrices [2x2, 4x4, 1x1] x hidden_size
            traj: trajectory point at current time-stamp (N x 2)
            timestamp_mask: (N x 1)
            same_scene_mask: [0,0,1,1,1,1,2]
        '''
        N = traj.size(0)
        all_scene_ids = torch.unique(same_scene_mask.long()).cpu().numpy().tolist()

        for scene_id in all_scene_ids:
            curr_ht = ht_list[scene_id]
            curr_ct = ct_list[scene_id]
            curr_mask = (same_scene_mask[:,0] == scene_id)
            # (curr_N x 2)
            curr_traj = traj[curr_mask==1, :]
            curr_N = curr_traj.size(0)
            # (curr_N x 1) -- e.g. [1, 1, 1, 0, 1]
            curr_timestamp_mask = (timestamp_mask[curr_mask, 0] == 1)
            # (curr_N) -- e.g. [0, 1, 2, 4]
            curr_timestamp_idx = torch.nonzero(curr_timestamp_mask.view(-1)).view(-1)
            curr_existing_N = torch.sum(curr_timestamp_mask).item()

            # If there is less than or equal to one agent, then there is
            # no spatial edge.
            if curr_existing_N <= 1:
                continue

            # Select only those edges that are present at current timestamp
            # (curr_existing_N x curr_N x hidden_size)
            curr_ht_0 = torch.index_select(curr_ht, 0, curr_timestamp_idx)
            # (curr_existing_N x curr_existing_N x hidden_size)
            curr_ht_1 = torch.index_select(curr_ht_0, 1, curr_timestamp_idx)
            # (curr_existing_N x curr_N x hidden_size)
            curr_ct_0 = torch.index_select(curr_ct, 0, curr_timestamp_idx)
            # (curr_existing_N x curr_existing_N x hidden_size)
            curr_ct_1 = torch.index_select(curr_ct_0, 1, curr_timestamp_idx)
            # (curr_existing_N x 1 x 2)
            curr_xt_1 = curr_traj[curr_timestamp_mask, :].view(curr_existing_N, 1, 2)
            # (curr_existing_N x curr_existing_N x 2)
            curr_xt_1 = torch.transpose(curr_xt_1.repeat(1, curr_existing_N, 1), 0, 1).float() -\
                        curr_xt_1.repeat(1, curr_existing_N, 1).float()

            # Process with LSTM.
            # (curr_existing_N**2, embed_size)
            e_uv = self.embed(curr_xt_1.view(curr_existing_N**2, 2))
            _, (curr_ht_1, curr_ct_1) = self.lstm(e_uv.view(curr_existing_N**2, 1, -1), \
                (curr_ht_1.view(1, curr_existing_N**2, -1), curr_ct_1.view(1, curr_existing_N**2, -1)))
            curr_ht_1 = curr_ht_1.view(curr_existing_N, curr_existing_N, -1)
            curr_ct_1 = curr_ct_1.view(curr_existing_N, curr_existing_N, -1)

            # Scatter back to the original matrices.
            curr_ht_0 = curr_ht_0.scatter(
                1, curr_timestamp_idx.view(1,curr_existing_N,1).repeat(curr_existing_N,1,self.hidden_size), curr_ht_1)
            curr_ht = curr_ht.scatter(
                0, curr_timestamp_idx.view(curr_existing_N,1,1).repeat(1,curr_N,self.hidden_size), curr_ht_0)
            curr_ct_0 = curr_ct_0.scatter(
                1, curr_timestamp_idx.view(1,curr_existing_N,1).repeat(curr_existing_N,1,self.hidden_size), curr_ct_1)
            curr_ct = curr_ct.scatter(
                0, curr_timestamp_idx.view(curr_existing_N,1,1).repeat(1,curr_N,self.hidden_size), curr_ct_0)

            # Update the list of ht/ct:
            ht_list[scene_id] = curr_ht
            ct_list[scene_id] = curr_ct

        return ht_list, ct_list


class TemporalEdgeRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=256):
        super(TemporalEdgeRNN, self).__init__()
        self.embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, ht, ct, xt):
        '''The forward function for temporal-edge RNN.

            ht: (N x hidden_size)
            ct: (N x hidden_size)
            xt: (N x 2)
        '''
        N = ht.size(0)
        # (N x embed_size)
        e_vv = self.embed(xt)
        _, (ht_new, ct_new) = self.lstm(
            e_vv.view(N, 1, -1), (ht.view(1, N, -1), ct.view(1, N, -1)))
        ht_new = ht_new.view(N, -1)
        ct_new = ct_new.view(N, -1)

        return ht_new, ct_new


class EdgeToNodeAttention(nn.Module):
    def __init__(self):
        return

    def forward(self, X):
        return X


class NodeRNN(nn.Module):
    def __init__(self):
        return

    def forward(self, X):
        return X


class ProbablisticTrajectoryLoss:
    def loss_fn(self, y_pred_tuple, y_true):
        y_pred, y_traj = y_pred_tuple
        if y_pred is None:
            return 0
        # y_pred: N x pred_len x 5
        # y_true: (pred_traj, pred_traj_rel)  N x pred_len x 2
        mux, muy, sigma_x, sigma_y, corr = y_pred[:,:,0], y_pred[:,:,1],\
            y_pred[:,:,2], y_pred[:,:,3], y_pred[:,:,4]
        is_predictable = y_true[2].long()
        x, y = y_true[1][is_predictable[:,0]==1,:,0].float(), \
               y_true[1][is_predictable[:,0]==1,:,1].float()
        N = y_pred.size(0)
        if N == 0:
            return 0

        eps = 1e-10

        z = ((x-mux)/(eps+sigma_x))**2 + ((y-muy)/(eps+sigma_y))**2 - \
            2*corr*(x-mux)*(y-muy)/(sigma_x*sigma_y+eps)
        P = 1/(2*np.pi*sigma_x*sigma_y*torch.sqrt(1-corr**2)+eps) * \
            torch.exp(-z/(2*(1-corr**2)))

        loss = torch.clamp(P, min=eps)
        loss = -loss.log()
        return torch.sum(loss)/N

    def loss_info(self, y_pred_tuple, y_true):
        y_pred, y_pred_traj = y_pred_tuple
        is_predictable = y_true[2].long()

        # loss = nn.MSELoss()
        # out = loss(y_pred[:, :, :2], y_true[1][is_predictable[:,0]==1,:,:].float())
        # return out

        loss = nn.MSELoss()
        #y_pred_traj = y_true[0][is_predictable[:,0]==1,:,:].float()
        #for i in range(1, y_pred_traj.size(1)):
        #    y_pred_traj[:, i, :] = y_pred_traj[:, i-1, :] + y_pred[:, i, :2]
        out = loss(y_pred_traj, y_true[0][is_predictable[:,0]==1,:,:].float())
        return out

        # out = y_pred[:, :, :2].float() - y_true[1][is_predictable[:,0]==1,:,:].float()
        # out = out ** 2
        # out = torch.sum(out, 2)
        # out = torch.sqrt(out)
        # out = torch.mean(out)
        # return out
        