#!/usr/bin/env python

import torch
import torch.nn as nn

from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *
from learning_algorithms.prediction.models.lane_attention_trajectory_model.coord_conversion_utils import *


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
            disp_embedding = self.disp_embed(
                (curr_obs_pos_rel[ts_obs_mask, :]).clone()).view(curr_N, 1, -1)
            _, (ht_new, ct_new) = self.lstm(
                disp_embedding, (ht[ts_obs_mask, :].view(1, curr_N, -1), ct[ts_obs_mask, :].view(1, curr_N, -1)))
            ht[ts_obs_mask, :] = ht_new.view(curr_N, -1)
            ct[ts_obs_mask, :] = ct_new.view(curr_N, -1)

        return pred_out, pred_traj


class FixedLane_LSTM(nn.Module):
    def __init__(self, pred_len=29,
                 obs_embed_size=32, obs_hidden_size=64,
                 lane_embed_size=32, lane_hidden_size=64,
                 all_embed_size=256, all_hidden_size=256,
                 lane_info_enc_size=64,
                 pred_mlp_param=[64, 5]):
        super(FixedLane_LSTM, self).__init__()
        self.pred_len = pred_len

        # For obstacle LSTM encoding.
        self.obs_embed = torch.nn.Sequential(
            nn.Linear(2, obs_embed_size),
            nn.ReLU(),
        )
        self.obs_h0, self.obs_c0 = generate_lstm_states(obs_hidden_size)
        self.obs_lstm = nn.LSTM(obs_embed_size, obs_hidden_size, num_layers=1, batch_first=True)

        # For lane LSTM encoding.
        self.lane_h0, self.lane_c0 = generate_lstm_states(lane_hidden_size)
        self.obs2lane_lstm = ObstacleToLaneEncoding(
            embed_size=lane_embed_size, hidden_size=lane_hidden_size, mode=0)

        # For future lane info encoding.
        self.lane_future_encode = LaneFutureEncoding()
        self.lane_info_encode = torch.nn.Sequential(
            nn.Linear(2, lane_info_enc_size),
            nn.ReLU(),
        )

        # Lane-Pooling.
        self.filter_lane = LanePoolingSimple()

        # For overall LSTM encoding.
        self.all_h0, self.all_c0 = generate_lstm_states(all_hidden_size)
        self.all_lstm = nn.LSTM(all_embed_size, all_hidden_size, num_layers=1, batch_first=True)

        # For output prediction
        self.pred_layer = generate_mlp([all_hidden_size] + pred_mlp_param,
                                       dropout=0.0, last_layer_nonlinear=False)

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_pos: N x 20 x 2
            - obs_pos_rel: N x 20 x 2
            - lane_features: M x 150 x 4
            - same_obstacle_mask: M x 1
        '''
        obs_hist_size, obs_pos, obs_pos_rel, lane_features, same_obstacle_mask = X
        N = obs_pos.size(0)
        M = lane_features.size(0)
        observation_len = obs_pos.size(1)
        obs_ht, obs_ct = self.obs_h0.repeat(N, 1), self.obs_c0.repeat(N, 1)
        lane_ht, lane_ct = self.lane_h0.repeat(M, 1), self.lane_c0.repeat(M, 1)
        all_ht, all_ct = self.all_h0.repeat(N, 1), self.all_c0.repeat(N, 1)
        Ht = None

        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))
        this_timestamp_mask, this_obs_pos, this_obs_pos_rel = None, None, None

        # Do lane LSTM.
        # (M x 64), (M x 64), (M x 2)
        dummy1, dummy2, lane_dist, dummy3, dummy4 = self.obs2lane_lstm(lane_features, obs_pos[:, observation_len-1, :].float(),
                                                                       same_obstacle_mask, pred_mask, lane_ht, lane_ct)

        for t in range(1, observation_len+self.pred_len):
            # Select the proper input data
            if t < observation_len:
                this_timestamp_mask = (obs_hist_size > observation_len-t).long().view(-1)
                this_obs_pos_rel = obs_pos_rel[:, t, :].float()
                this_obs_pos = obs_pos[:, t, :].float()
            else:
                this_timestamp_mask = pred_mask
                pred_out[:, t-observation_len, :] = self.pred_layer(Ht.clone()).float().clone()
                this_obs_pos_rel = pred_out[:, t-observation_len, :2]
                this_obs_pos = this_obs_pos + this_obs_pos_rel
                pred_traj[:, t-observation_len, :] = this_obs_pos.clone()

            curr_N = torch.sum(this_timestamp_mask).long().item()
            if curr_N == 0:
                continue
            this_timestamp_mask = (this_timestamp_mask == 1)

            # Do obstacle LSTM.
            obs_embedding = self.obs_embed(
                (this_obs_pos_rel[this_timestamp_mask, :]).clone()).view(curr_N, 1, -1)
            _, (obs_ht_new, obs_ct_new) = self.obs_lstm(obs_embedding,
                                                        (obs_ht[this_timestamp_mask, :].view(1, curr_N, -1), obs_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            # (N x 64)
            obs_ht[this_timestamp_mask, :] = obs_ht_new.view(curr_N, -1)
            obs_ct[this_timestamp_mask, :] = obs_ct_new.view(curr_N, -1)

            # Do lane LSTM.
            # (M x 64), (M x 64), (M x 2)
            lane_ht, lane_ct, dummy3, dummy1, dummy2 = self.obs2lane_lstm(lane_features, this_obs_pos,
                                                                          same_obstacle_mask, this_timestamp_mask, lane_ht, lane_ct)

            # Select lane of interest and encode.
            lane_idx_of_interest = self.filter_lane(torch.sum(lane_dist**2, 1),
                                                    same_obstacle_mask, this_timestamp_mask)

            # (curr_N x 2)
            lane_dist_new = lane_dist[lane_idx_of_interest, :]
            # (curr_N x 64)
            lane_info_enc = self.lane_info_encode(lane_dist_new)
            # (curr_N x 64)
            lane_future_enc = self.lane_future_encode(
                lane_features[lane_idx_of_interest, :, :], this_obs_pos[this_timestamp_mask, :])
            # (curr_N x 64)
            lane_ht_of_interest = lane_ht[lane_idx_of_interest, :].view(curr_N, -1)

            # Get the overall encoding.
            # (curr_N x 256)
            all_enc = torch.cat((obs_ht[this_timestamp_mask, :],
                                 lane_ht_of_interest, lane_info_enc, lane_future_enc), 1)

            _, (all_ht_new, all_ct_new) = self.all_lstm(all_enc.view(curr_N, 1, -1),
                                                        (all_ht[this_timestamp_mask, :].view(1, curr_N, -1), all_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            all_ht[this_timestamp_mask, :] = all_ht_new.view(curr_N, -1)
            all_ct[this_timestamp_mask, :] = all_ct_new.view(curr_N, -1)
            Ht = all_ht

        return pred_out, pred_traj


class LanePooling_LSTM(nn.Module):
    def __init__(self, pred_len=29,
                 obs_embed_size=32, obs_hidden_size=64,
                 lane_embed_size=32, lane_hidden_size=64,
                 all_embed_size=256, all_hidden_size=256,
                 lane_info_enc_size=64,
                 pred_mlp_param=[64, 5]):
        super(LanePooling_LSTM, self).__init__()
        self.pred_len = pred_len

        # For obstacle LSTM encoding.
        self.obs_embed = torch.nn.Sequential(
            nn.Linear(2, obs_embed_size),
            nn.ReLU(),
        )
        self.obs_h0, self.obs_c0 = generate_lstm_states(obs_hidden_size)
        self.obs_lstm = nn.LSTM(obs_embed_size, obs_hidden_size, num_layers=1, batch_first=True)

        # For lane LSTM encoding.
        self.lane_h0, self.lane_c0 = generate_lstm_states(lane_hidden_size)
        self.obs2lane_lstm = ObstacleToLaneEncoding(
            embed_size=lane_embed_size, hidden_size=lane_hidden_size, mode=0)

        # For future lane info encoding.
        self.lane_future_encode = LaneFutureEncoding()
        self.lane_info_encode = torch.nn.Sequential(
            nn.Linear(2, lane_info_enc_size),
            nn.ReLU(),
        )

        # Lane-Pooling.
        self.filter_lane = LanePoolingSimple()

        # For overall LSTM encoding.
        self.all_h0, self.all_c0 = generate_lstm_states(all_hidden_size)
        self.all_lstm = nn.LSTM(all_embed_size, all_hidden_size, num_layers=1, batch_first=True)

        # For output prediction
        self.pred_layer = generate_mlp([all_hidden_size] + pred_mlp_param,
                                       dropout=0.0, last_layer_nonlinear=False)

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_pos: N x 20 x 2
            - obs_pos_rel: N x 20 x 2
            - lane_features: M x 150 x 4
            - same_obstacle_mask: M x 1
        '''
        obs_hist_size, obs_pos, obs_pos_rel, lane_features, same_obstacle_mask = X
        N = obs_pos.size(0)
        M = lane_features.size(0)
        observation_len = obs_pos.size(1)
        obs_ht, obs_ct = self.obs_h0.repeat(N, 1), self.obs_c0.repeat(N, 1)
        lane_ht, lane_ct = self.lane_h0.repeat(M, 1), self.lane_c0.repeat(M, 1)
        all_ht, all_ct = self.all_h0.repeat(N, 1), self.all_c0.repeat(N, 1)
        Ht = None

        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))
        this_timestamp_mask, this_obs_pos, this_obs_pos_rel = None, None, None
        for t in range(1, observation_len+self.pred_len):
            # Select the proper input data
            if t < observation_len:
                this_timestamp_mask = (obs_hist_size > observation_len-t).long().view(-1)
                this_obs_pos_rel = obs_pos_rel[:, t, :].float()
                this_obs_pos = obs_pos[:, t, :].float()
            else:
                this_timestamp_mask = pred_mask
                pred_out[:, t-observation_len, :] = self.pred_layer(Ht.clone()).float().clone()
                this_obs_pos_rel = pred_out[:, t-observation_len, :2]
                this_obs_pos = this_obs_pos + this_obs_pos_rel
                pred_traj[:, t-observation_len, :] = this_obs_pos.clone()

            curr_N = torch.sum(this_timestamp_mask).long().item()
            if curr_N == 0:
                continue
            this_timestamp_mask = (this_timestamp_mask == 1)

            # Do obstacle LSTM.
            obs_embedding = self.obs_embed(
                (this_obs_pos_rel[this_timestamp_mask, :]).clone()).view(curr_N, 1, -1)
            _, (obs_ht_new, obs_ct_new) = self.obs_lstm(obs_embedding,
                                                        (obs_ht[this_timestamp_mask, :].view(1, curr_N, -1), obs_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            # (N x 64)
            obs_ht[this_timestamp_mask, :] = obs_ht_new.view(curr_N, -1)
            obs_ct[this_timestamp_mask, :] = obs_ct_new.view(curr_N, -1)

            # Do lane LSTM.
            # (M x 64), (M x 64), (M x 2)
            lane_ht, lane_ct, lane_dist, dummy1, dummy2 = self.obs2lane_lstm(lane_features, this_obs_pos,
                                                                             same_obstacle_mask, this_timestamp_mask, lane_ht, lane_ct)

            # Select lane of interest and encode.
            lane_idx_of_interest = self.filter_lane(torch.sum(lane_dist**2, 1),
                                                    same_obstacle_mask, this_timestamp_mask)
            # (curr_N x 2)
            lane_dist_new = lane_dist[lane_idx_of_interest, :]
            # (curr_N x 64)
            lane_info_enc = self.lane_info_encode(lane_dist_new)
            # (curr_N x 64)
            lane_future_enc = self.lane_future_encode(
                lane_features[lane_idx_of_interest, :, :], this_obs_pos[this_timestamp_mask, :])
            # (curr_N x 64)
            lane_ht_of_interest = lane_ht[lane_idx_of_interest, :].view(curr_N, -1)

            # Get the overall encoding.
            # (curr_N x 256)
            all_enc = torch.cat((obs_ht[this_timestamp_mask, :],
                                 lane_ht_of_interest, lane_info_enc, lane_future_enc), 1)

            _, (all_ht_new, all_ct_new) = self.all_lstm(all_enc.view(curr_N, 1, -1),
                                                        (all_ht[this_timestamp_mask, :].view(1, curr_N, -1), all_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            all_ht[this_timestamp_mask, :] = all_ht_new.view(curr_N, -1)
            all_ct[this_timestamp_mask, :] = all_ct_new.view(curr_N, -1)
            Ht = all_ht

        return pred_out, pred_traj


class LaneAttention_LSTM(nn.Module):
    def __init__(self, pred_len=29, lane_future_mode=2, soft_argmax_degree=1.0,
                 obs_embed_size=32, obs_hidden_size=64,
                 lane_embed_size=32, lane_hidden_size=64,
                 all_embed_size=256, all_hidden_size=256,
                 lane_info_enc_size=64,
                 pred_mlp_param=[64, 5]):
        super(LaneAttention_LSTM, self).__init__()
        self.pred_len = pred_len
        self.lane_future_mode = lane_future_mode
        self.soft_argmax_degree = soft_argmax_degree

        # For obstacle LSTM encoding.
        self.obs_embed = torch.nn.Sequential(
            nn.Linear(2, obs_embed_size),
            nn.ReLU(),
        )
        self.obs_h0, self.obs_c0 = generate_lstm_states(obs_hidden_size)
        self.obs_lstm = nn.LSTM(obs_embed_size, obs_hidden_size, num_layers=1, batch_first=True)

        # For lane encoding.
        self.lane_h0, self.lane_c0 = generate_lstm_states(lane_hidden_size)
        if lane_future_mode == 0:
            self.obs2lane_lstm = ObstacleToLaneEncoding(mode=0)
        elif lane_future_mode == 1:
            self.obs2lane_lstm = ObstacleToLaneEncoding(
                embed_size=lane_embed_size, hidden_size=lane_hidden_size,
                info_enc_size=lane_hidden_size, future_enc_size=lane_hidden_size,
                mode=1)
        elif lane_future_mode == 2:
            self.obs2lane_lstm = ObstacleToLaneEncoding(
                embed_size=lane_embed_size, hidden_size=lane_hidden_size,
                info_enc_size=lane_hidden_size, future_enc_size=lane_hidden_size,
                mode=2)

        # For lane-aggregating
        if lane_future_mode == 0:
            self.aggregation = GetAggregatedLaneEnc(
                soft_argmax_degree=soft_argmax_degree, lane_info_enc_size=64, lane_future_enc_size=0, aggr_enc_size=128)
        elif lane_future_mode == 1:
            self.aggregation = GetAggregatedLaneEnc(
                soft_argmax_degree=soft_argmax_degree, lane_info_enc_size=64, lane_future_enc_size=64, aggr_enc_size=192)
        elif lane_future_mode == 2:
            self.aggregation = GetAggregatedLaneEnc(soft_argmax_degree=soft_argmax_degree)

        # For overall LSTM encoding.
        self.all_h0, self.all_c0 = generate_lstm_states(all_hidden_size)
        self.all_lstm = nn.LSTM(all_embed_size, all_hidden_size, num_layers=1, batch_first=True)

        # For output prediction
        self.pred_layer = generate_mlp([all_hidden_size] + pred_mlp_param,
                                       dropout=0.0, last_layer_nonlinear=False)

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_pos: N x 20 x 2
            - obs_pos_rel: N x 20 x 2
            - lane_features: M x 150 x 4
            - same_obstacle_mask: M x 1
        '''
        obs_hist_size, obs_pos, obs_pos_rel, lane_features, same_obstacle_mask = X
        N = obs_pos.size(0)
        M = lane_features.size(0)
        observation_len = obs_pos.size(1)
        obs_ht, obs_ct = self.obs_h0.repeat(N, 1), self.obs_c0.repeat(N, 1)
        lane_ht, lane_ct = self.lane_h0.repeat(M, 1), self.lane_c0.repeat(M, 1)
        all_ht, all_ct = self.all_h0.repeat(N, 1), self.all_c0.repeat(N, 1)
        Ht = None

        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 5))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))
        this_timestamp_mask, this_obs_pos, this_obs_pos_rel = None, None, None
        for t in range(1, observation_len+self.pred_len):
            # Select the proper input data
            if t < observation_len:
                this_timestamp_mask = (obs_hist_size > observation_len-t).long().view(-1)
                this_obs_pos_rel = obs_pos_rel[:, t, :].float()
                this_obs_pos = obs_pos[:, t, :].float()
            else:
                this_timestamp_mask = pred_mask
                pred_out[:, t-observation_len, :] = self.pred_layer(Ht.clone()).float().clone()
                this_obs_pos_rel = pred_out[:, t-observation_len, :2]
                this_obs_pos = this_obs_pos + this_obs_pos_rel
                pred_traj[:, t-observation_len, :] = this_obs_pos.clone()

            curr_N = torch.sum(this_timestamp_mask).long().item()
            if curr_N == 0:
                continue
            this_timestamp_mask = (this_timestamp_mask == 1)

            # Do obstacle LSTM.
            obs_embedding = self.obs_embed(
                (this_obs_pos_rel[this_timestamp_mask, :]).clone()).view(curr_N, 1, -1)
            _, (obs_ht_new, obs_ct_new) = self.obs_lstm(obs_embedding,
                                                        (obs_ht[this_timestamp_mask, :].view(1, curr_N, -1), obs_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            # (N x 64)
            obs_ht[this_timestamp_mask, :] = obs_ht_new.view(curr_N, -1)
            obs_ct[this_timestamp_mask, :] = obs_ct_new.view(curr_N, -1)

            # Do lane LSTM.
            # (M x 64), (M x 64), (curr_M x 64), (curr_M x 64), (M)
            lane_ht, lane_ct, lane_info_enc, lane_future_enc, lane_mask = self.obs2lane_lstm(
                lane_features, this_obs_pos, same_obstacle_mask, this_timestamp_mask, lane_ht, lane_ct)

            # Do lane_ht, lane_info_enc, and lane_future_enc attentional aggregation.
            if self.lane_future_mode == 0:
                lane_total_enc = self.aggregation(lane_ht[lane_mask, :], lane_info_enc,
                                                  _, same_obstacle_mask[lane_mask, :])
            else:
                lane_total_enc = self.aggregation(lane_ht[lane_mask, :], lane_info_enc,
                                                  lane_future_enc, same_obstacle_mask[lane_mask, :])

            # Get the overall encoding.
            # (curr_N x 256)
            all_enc = torch.cat((obs_ht[this_timestamp_mask, :], lane_total_enc), 1)

            # Go through overall-LSTM.
            _, (all_ht_new, all_ct_new) = self.all_lstm(all_enc.view(curr_N, 1, -1),
                                                        (all_ht[this_timestamp_mask, :].view(1, curr_N, -1), all_ct[this_timestamp_mask, :].view(1, curr_N, -1)))
            all_ht[this_timestamp_mask, :] = all_ht_new.view(curr_N, -1)
            all_ct[this_timestamp_mask, :] = all_ct_new.view(curr_N, -1)
            Ht = all_ht

        return pred_out, pred_traj


################################################################################
# Sub-Models
################################################################################

class ObstacleToLaneEncoding(nn.Module):
    def __init__(self, embed_size=32, hidden_size=64,
                 info_enc_size=64, future_enc_size=64,
                 mode=0):
        '''
        mode: 0 - no future encoding
              1 - MLP future encoding
              2 - RNN future encoding
        '''
        super(ObstacleToLaneEncoding, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.info_enc_size = info_enc_size
        self.future_enc_size = future_enc_size
        self.mode = mode

        self.get_proj_pt = ObstacleToLaneRelation()

        self.embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

        self.lane_info_encode = torch.nn.Sequential(
            nn.Linear(2, info_enc_size),
            nn.ReLU(),
        )

        if mode == 0:
            self.lane_future_encode = None
        elif mode == 1:
            self.lane_future_encode = LaneFutureEncoding(
                window_size=5, delta_s=1.0, rnn_encoding=False)
        elif mode == 2:
            self.lane_future_encode = LaneFutureEncoding()

    def forward(self, lane_features, obs_pos, same_obs_mask, ts_mask, ht, ct):
        '''
        params:
            - lane_features: M x 150 x 4
            - obs_pos: N x 2
            - same_obs_mask: M x 1
            - ts_mask: N
            - ht: M x hidden_size
            - ct: M x hidden_size
        return:
            - ht_return: M x hidden_size
            - ct_return: M x hidden_size
            - lane_info_enc: curr_M x info_enc_size
            - lane_future_enc: curr_M x future_enc_size
        '''
        N = obs_pos.size(0)
        M = same_obs_mask.size(0)

        # Mask processing
        # (M x N)
        same_obs_mask_repeated = same_obs_mask.repeat(
            1, same_obs_mask.max().long().item() + 1).float()
        incremental_mask = cuda(torch.ones(1, same_obs_mask.max().long().item()))
        incremental_mask = torch.cumsum(incremental_mask, dim=1)
        incremental_mask = torch.cat((cuda(torch.zeros(1, 1)), incremental_mask), 1)
        # (M x N)
        incremental_mask = incremental_mask.repeat(M, 1).float()
        lane_mask = ts_mask.view(1, N).repeat(M, 1).long() * \
            (incremental_mask == same_obs_mask_repeated).long()
        lane_mask = torch.sum(lane_mask, 1)
        curr_M = torch.sum(lane_mask).long().item()
        lane_mask = (lane_mask == 1)

        # Do obstacle-to-lane encoding.
        # (curr_M x 2), (curr_M x 2), (curr_M x 2)
        proj_pt, indices, repeated_obs_pos = self.get_proj_pt(lane_features, obs_pos, same_obs_mask)
        rel_pos = proj_pt[lane_mask, :] - repeated_obs_pos[lane_mask, :]
        et = self.embed(rel_pos).view(curr_M, 1, -1)
        _, (ht_new, ct_new) = self.lstm(et, (ht[lane_mask, :].view(1, curr_M, -1),
                                             ct[lane_mask, :].view(1, curr_M, -1)))
        ht_return = ht.clone()
        ct_return = ct.clone()
        # (M x 64)
        ht_return[lane_mask, :] = ht_new
        ct_return[lane_mask, :] = ct_new

        if self.mode == 0:
            lane_info_enc = cuda(torch.zeros(M, 2))
            lane_info_enc[lane_mask, :] = rel_pos
            lane_future_enc = None

        elif self.mode == 1:
            # (curr_M x 64)
            lane_info_enc = self.lane_info_encode(rel_pos)
            # (curr_M x 64)
            lane_future_enc = self.lane_future_encode(
                lane_features[lane_mask, :, :], repeated_obs_pos[lane_mask, :])
        elif self.mode == 2:
            # Do lane-info encoding.
            # (curr_M x 64)
            lane_info_enc = self.lane_info_encode(rel_pos)
            # Do lane-future encoding.
            # (curr_M x 64)
            lane_future_enc = self.lane_future_encode(
                lane_features[lane_mask, :, :], repeated_obs_pos[lane_mask, :])

        return ht_return, ct_return, lane_info_enc, lane_future_enc, lane_mask


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
        h0, c0 = self.h0.view(1, 1, -1).repeat(1, N, 1), self.c0.view(1, 1, -1).repeat(1, N, 1)
        # (N x num_lane_pt x hidden_size)
        lane_states, _ = self.lstm(lane_embed, (h0, c0))

        # (N x hidden_size)
        return lane_states[:, -1, :]


class LaneFutureEncoding(nn.Module):
    # TODO(jiacheng): make it output the future-lane-points, without any encodings.
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
        proj_pt, _ = self.get_projection_point(
            lane_features[torch.arange(N), idx_before, :2], lane_features[torch.arange(N), idx_after, :2], obs_pos)

        # Get obstalces' SL-coord
        # (N x 2)
        sl_coord = self.proj_pt_to_sl(proj_pt, proj_pt-obs_pos, idx_before,
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
        xy_coord = self.sl_to_xy(
            lane_features.view(N, 1, 150, 4).repeat(1, self.window_size+1,
                                                    1, 1).view(N*(self.window_size+1), 150, 4),
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


class LanePoolingSimple(nn.Module):
    def __init__(self):
        super(LanePoolingSimple, self).__init__()

    def forward(self, lane_dist, same_obstacle_mask, ts_mask):
        '''
        params:
            - lane_dist: M
            - same_obstacle_mask: M x 1
            - ts_mask: N
        return:
            - lane_of_interest_mask: M (sum of it = sum of ts_mask)
        '''
        M = lane_dist.size(0)
        N = ts_mask.size(0)

        # (M)
        lane_of_interest_mask = cuda(torch.zeros(M))
        num_visited_lanes = cuda(torch.zeros(1).long())
        # Go through every obstacle_id.
        for obs_id in range(same_obstacle_mask.max().long().item() + 1):
            # (curr_num_lane)
            curr_mask = (same_obstacle_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long()
            # Check if the obstacle is present yet or not.
            if ts_mask[obs_id] > 0:
                # Select the lane of interest based on the smallest distance.
                min_idx = torch.argmin(lane_dist[curr_mask])
                lane_of_interest_mask[num_visited_lanes+min_idx] = cuda(torch.ones(1))
            num_visited_lanes = num_visited_lanes + curr_num_lane
        lane_of_interest_mask = (lane_of_interest_mask > 0)

        return lane_of_interest_mask


class GetAggregatedLaneEnc(nn.Module):
    def __init__(self, soft_argmax_degree=1.0,
                 lane_hidden_size=64, lane_info_enc_size=64,
                 lane_future_enc_size=64, aggr_enc_size=192):
        super(GetAggregatedLaneEnc, self).__init__()
        self.soft_argmax_degree = soft_argmax_degree
        self.lane_hidden_size = lane_hidden_size
        self.lane_info_enc_size = lane_info_enc_size
        self.lane_future_enc_size = lane_future_enc_size
        self.aggr_enc_size = aggr_enc_size

        self.lane_scoring_mlp = generate_mlp(
            [lane_hidden_size+lane_info_enc_size] + [16, 1], dropout=0.0, last_layer_nonlinear=False)
        self.softmax_layer = nn.Softmax()

    def forward(self, lane_ht, lane_info_enc, lane_future_enc, same_obstacle_mask):
        '''
        params:
            - lane_ht: M x hidden_size
            - lane_info_enc: M x info_enc_size
            - lane_future_enc: M x future_enc_size
            - same_obstacle_mask: M x 1
        return:
            - lane_aggr_enc: N x aggr_enc_size
        '''
        M = lane_ht.size(0)
        N = torch.unique(same_obstacle_mask).size(0)

        lane_aggr_enc = cuda(torch.zeros(N, self.aggr_enc_size))
        count = 0

        # (M x N)
        same_obs_mask_repeated = same_obstacle_mask.repeat(1, N).float()
        # (1 x N)
        id_mask = torch.unique(same_obstacle_mask).view(1, N)
        # (M x N)
        id_mask = id_mask.repeat(M, 1).float()
        # (M x N)
        special_mask = (id_mask == same_obs_mask_repeated).float()

        # (M x (hidden_size + info_enc_size))
        lane_score_input = torch.cat((lane_ht, lane_info_enc), 1)
        # (M x 1)
        lane_score = self.lane_scoring_mlp(lane_score_input).view(M, 1)
        lane_score = lane_score * self.soft_argmax_degree
        # (M x N)
        lane_score = lane_score.repeat(1, N) * special_mask
        anti_special_mask = -1 * (special_mask - 1)
        lane_score_min, _ = torch.min(lane_score, 0)
        lane_score_min = lane_score_min.view(1, N).repeat(M, 1)
        lane_score = lane_score + lane_score_min * anti_special_mask
        lane_score_max, _ = torch.max(lane_score, 0)
        lane_score_max = lane_score_max.view(1, N).repeat(M, 1)
        lane_score = (lane_score - lane_score_max) * special_mask
        lane_score_numerator = torch.exp(lane_score) * special_mask
        lane_score_denominator = torch.sum(lane_score_numerator, 0).view(1, N).repeat(M, 1)
        # (M x N)
        lane_prob = lane_score_numerator / lane_score_denominator * special_mask

        if self.lane_future_enc_size != 0:
            lane_total_enc = torch.cat((lane_ht, lane_info_enc, lane_future_enc), 1)
        else:
            lane_total_enc = torch.cat((lane_ht, lane_info_enc), 1)

        lane_total_enc_size = lane_total_enc.size(1)
        lane_total_enc = lane_total_enc.view(M, 1, lane_total_enc_size).repeat(1, N, 1)
        lane_total_enc = lane_total_enc * lane_prob.view(M, N, 1).repeat(1, 1, lane_total_enc_size)
        # (N x lane_total_enc_size)
        lane_total_enc = torch.sum(lane_total_enc, 0)

        return lane_total_enc


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
        mux, muy, sigma_x, sigma_y, corr = y_pred[:, :, 0], y_pred[:, :, 1],\
            y_pred[:, :, 2], y_pred[:, :, 3], y_pred[:, :, 4]
        is_predictable = y_true[2].long()
        x, y = y_true[1][is_predictable[:, 0] == 1, :, 0].float(), \
            y_true[1][is_predictable[:, 0] == 1, :, 1].float()
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

        out = loss(y_pred_traj, y_true[0][is_predictable[:, 0] == 1, 1:, :].float())
        return out
