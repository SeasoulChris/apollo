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


################################################################################
# Overall Big Models
################################################################################

class SelfLSTM(nn.Module):
    def __init__(self, pred_len=29, embed_size=64, hidden_size=128):
        super(SelfLSTM, self).__init__()
        self.pred_len = pred_len
        self.disp_embed_size = embed_size
        self.hidden_size = hidden_size

        self.disp_embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size, 5),
        )

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_pos: N x 20 x 2
            - obs_pos_rel: N x 20 x 2
            - lane_features: M x 150 x 4
            - same_obs_mask: M x 1
        '''
        obs_hist_size, obs_pos, obs_pos_rel, lane_features, same_obs_mask = X
        N = obs_pos.size(0)
        observation_len = obs_pos.size(1)
        ht, ct = self.h0.repeat(N, 1), self.h0.repeat(N, 1)

        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))
        ts_obs_mask, curr_obs_pos, curr_obs_pos_rel = None, None, None
        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                ts_obs_mask = (obs_hist_size > observation_len-t).long().view(-1)
                curr_obs_pos_rel = obs_pos_rel[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                ts_obs_mask = pred_mask
                pred_out[:, t-observation_len, :] = self.pred_layer(ht.clone()).float().clone()
                curr_obs_pos_rel = pred_out[:, t-observation_len, :2]
                curr_obs_pos = curr_obs_pos + curr_obs_pos_rel
                pred_traj[:, t-observation_len, :] = curr_obs_pos.clone()

            curr_N = torch.sum(ts_obs_mask).long().item()
            if curr_N == 0:
                continue

            ts_obs_mask = (ts_obs_mask == 1)
            disp_embedding = self.disp_embed((curr_obs_pos_rel[ts_obs_mask,:]).clone()).view(curr_N, 1, -1)
            _, (ht_new, ct_new) = self.lstm(
            	disp_embedding, (ht[ts_obs_mask, :].view(1,curr_N,-1), ct[ts_obs_mask, :].view(1,curr_N,-1)))
            ht[ts_obs_mask, :] = ht_new.view(curr_N, -1)
            ct[ts_obs_mask, :] = ct_new.view(curr_N, -1)

        return pred_out, pred_traj


################################################################################
# Sub-Models
################################################################################

class LaneLSTM(nn.Module):
    def __init__(self, embed_size=32, hidden_size=64):
        super(LaneLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, lane_points):
        '''
        params:
            - lane_points: N x num_lane_pt x 2
        return:
            - encoding: N x hidden_size
        '''
        N = lane_points.size(0)
        num_lane_pt = lane_points.size(1)

        # Embed the lane-points.
        # (N*num_lane_pt x 2)
        lane_embed_input = lane_points.view(N*num_lane_pt, 2)
        # (N x num_lane_pt x embed_size)
        lane_embed = self.embed(lane_embed_input).view(N, num_lane_pt, -1)

        # Go through RNN.
        h0, c0 = self.h0.view(1,1,-1).repeat(1, N, 1), self.c0.view(1,1,-1).repeat(1, N, 1)
        # (N x num_lane_pt x hidden_size)
        lane_states, _ = self.lstm(lane_embed, (h0, c0))

        # (N x hidden_size)
        return lane_states[:, -1, :]


class LaneFutureEncoding(nn.Module):
    '''
    Given a lane and an obstacle, first extract the lane's future-shape
    starting from the obstacle's position, then encode it as the lane's
    future-encoding.
    '''
    def __init__(self, window_size=10, delta_s=0.5,
                 rnn_encoding=True, debug_mode=False):
        super(LaneFutureEncoding, self).__init__()
        self.window_size = window_size
        self.delta_s = delta_s
        self.rnn_encoding = rnn_encoding
        self.debug_mode = debug_mode

        self.find_the_closest_two_points = FindClosestLineSegmentFromLineToPoint()
        self.get_projection_point = PointToLineProjection()
        self.proj_pt_to_sl = ProjPtToSL()
        self.sl_to_xy = SLToXY()

        if rnn_encoding:
            self.lane_lstm = LaneLSTM(embed_size=32, hidden_size=64)
        else:
            self.lane_enc = torch.nn.Sequential(
                nn.Linear(window_size*2, 64),
                nn.ReLU(),
            )

    def forward(self, lane_features, obs_pos):
        '''
        params:
            - selected lane_features of interest: N x 150 x 4
            - obs_pos: N x 2
        return:
            - encoded lane future: N x enc_size
        '''
        N = obs_pos.size(0)
        # Get lane's relative distance to obstacle
        idx_before, idx_after = self.find_the_closest_two_points(lane_features, obs_pos)
        proj_pt, _ = self.get_projection_point(\
            lane_features[torch.arange(N),idx_before,:2], lane_features[torch.arange(N),idx_after,:2], obs_pos)

        # Get obstalces' SL-coord
        # (N x 2)
        sl_coord = self.proj_pt_to_sl(proj_pt, proj_pt-obs_pos, idx_before,\
            idx_after, lane_features)

        # Increment every delta_s=0.5 and get a new sequence of SL
        sl_coord[:, 1] = cuda(torch.zeros(N))
        sl_coord = sl_coord.view(N, 1, 2)
        # (N x 2)
        delta_sl = torch.cat((cuda(torch.ones(N, 1))*self.delta_s, cuda(torch.zeros(N, 1))), 1)
        # (N x window_size+1 x 2)
        delta_sl = delta_sl.view(N, 1, 2).repeat(1, self.window_size+1, 1)
        delta_sl[:, 0, :] = cuda(torch.zeros(N, 2))
        cum_sl = torch.cumsum(delta_sl, 1)
        # (N x window_size+1 x 2)
        sl_coord_new = sl_coord.repeat(1, self.window_size+1, 1) + cum_sl

        # Convert that SL back to XY and get delta_X, delta_Y
        xy_coord = self.sl_to_xy(\
            lane_features.view(N, 1, 150, 4).repeat(1, self.window_size+1, 1, 1).view(N*(self.window_size+1), 150, 4), \
            sl_coord_new.view(N*(self.window_size+1), 2))
        if self.debug_mode:
            return xy_coord
        # (N x window_size+1 x 2)
        xy_coord = xy_coord.view(N, self.window_size+1, 2)
        # (N x window_size x 2)
        delta_xy = xy_coord[:, 1:, :] - xy_coord[:, :-1, :]

        if self.rnn_encoding:
            # Feed that delta_X and delta_Y through RNN and get encoding
            # (N x enc_size)
            enc = self.lane_lstm(delta_xy)
            return enc
        else:
            # (N x window_size*2)
            enc = self.lane_enc(delta_xy.view(N, -1))
            return enc


################################################################################
# Loss Functions
################################################################################

class ProbablisticTrajectoryLoss:
    def loss_fn(self, y_pred_tuple, y_true):
        y_pred, y_traj = y_pred_tuple
        if y_pred is None:
            return cuda(torch.tensor(0))
        # y_pred: N x pred_len x 5
        # y_true: (pred_traj, pred_traj_rel)  N x pred_len x 2
        mux, muy, sigma_x, sigma_y, corr = y_pred[:,:,0], y_pred[:,:,1],\
            y_pred[:,:,2], y_pred[:,:,3], y_pred[:,:,4]
        is_predictable = y_true[2].long()
        x, y = y_true[1][is_predictable[:,0]==1,:,0].float(), \
               y_true[1][is_predictable[:,0]==1,:,1].float()
        N = y_pred.size(0)
        if N == 0:
            return cuda(torch.tensor(0))

        eps = 1e-4

        corr = torch.clamp(corr, min=-1+eps, max=1-eps)
        z = (x-mux)**2/(sigma_x**2+eps) + (y-muy)**2/(sigma_y**2+eps) - \
            2*corr*(x-mux)*(y-muy)/(torch.sqrt((sigma_x*sigma_y)**2)+eps)
        z = torch.clamp(z, min=eps)

        P = 1/(2*np.pi*torch.sqrt((sigma_x*sigma_y)**2)*torch.sqrt(1-corr**2)+eps) \
            * torch.exp(-z/(2*(1-corr**2)))

        loss = torch.clamp(P, min=eps)
        loss = -loss.log()

        return torch.sum(loss)/N

    def loss_info(self, y_pred_tuple, y_true):
        y_pred, y_pred_traj = y_pred_tuple
        is_predictable = y_true[2].long()

        loss = nn.MSELoss()

        out = loss(y_pred_traj, y_true[0][is_predictable[:,0]==1,1:,:].float())
        return out
