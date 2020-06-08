#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from fueling.learning.network_utils import *
from fueling.learning.train_utils import *


class CruiseMLP(nn.Module):
    def __init__(self):
        super(CruiseMLP, self).__init__()
        self.vehicle_encoding = VehicleLSTM(embed_size=64, hidden_size=128, encode_size=128)
        self.lane_encoding = LaneLSTM(embed_size=64, hidden_size=128, encode_size=127)
        self.prediction_layer = DistributionalScoring(
            obs_enc_size=128, lane_enc_size=128, aggr_enc_size=1, mlp_size=[100, 16, 1])

    def forward(self, X):
        obs_hist_size, obs_features, backward_lane_points, lane_features, is_self_lane, same_obs_mask = X
        obs_hist_size = obs_hist_size.float()
        obs_features = obs_features.float()
        lane_features = lane_features.float()
        same_obs_mask = same_obs_mask.float()
        N = obs_features.size(0)
        M = lane_features.size(0)

        obs_enc = self.vehicle_encoding(obs_features)

        lane_enc_1 = self.lane_encoding(lane_features)
        lane_enc_2 = lane_features[:, 0].view(M, 1)
        lane_enc = torch.cat((lane_enc_1, lane_enc_2), 1)

        aggr_enc = cuda(torch.zeros(N, 1))

        out = self.prediction_layer(obs_enc, lane_enc, aggr_enc, same_obs_mask)

        return out


class FastLaneAttention(nn.Module):
    def __init__(self, mode=1):
        super(FastLaneAttention, self).__init__()
        self.mode = mode
        self.vehicle_encoding = VehicleLSTM(embed_size=64, hidden_size=128, encode_size=128)
        self.lane_encoding = LaneLSTM(embed_size=64, hidden_size=128, encode_size=127)
        self.lane_aggregate = SimpleAggregation()
        if mode == 0:
            self.prediction_layer = DistributionalScoring(
                obs_enc_size=128, lane_enc_size=128, aggr_enc_size=1, mlp_size=[100, 16, 1])
        elif mode == 1:
            self.prediction_layer = DistributionalScoring(
                obs_enc_size=128, lane_enc_size=128, aggr_enc_size=256, mlp_size=[100, 16, 1])

    def forward(self, X):
        obs_hist_size, obs_features, backward_lane_points, lane_features, is_self_lane, same_obs_mask = X
        obs_hist_size = obs_hist_size.float()
        obs_features = obs_features.float()
        lane_features = lane_features.float()
        same_obs_mask = same_obs_mask.float()
        N = obs_features.size(0)
        M = lane_features.size(0)

        obs_enc = self.vehicle_encoding(obs_features)

        lane_enc_1 = self.lane_encoding(lane_features)
        lane_enc_2 = lane_features[:, 0].view(M, 1)
        lane_enc = torch.cat((lane_enc_1, lane_enc_2), 1)

        aggr_enc = cuda(torch.zeros(N, 1))
        if self.mode >= 1:
            aggr_enc = self.lane_aggregate(obs_enc, lane_enc, same_obs_mask)

        out = self.prediction_layer(obs_enc, lane_enc, aggr_enc, same_obs_mask)

        return out


class LaneAttention(nn.Module):
    def __init__(self):
        super(LaneAttention, self).__init__()
        self.vehicle_encoding = VehicleDynamicLSTM(embed_size=64, hidden_size=128, encode_size=128)
        self.backward_lane_encoding = BackwardLaneLSTM(
            embed_size=16, hidden_size=64, encode_size=64)
        self.lane_encoding = AttentionalLaneLSTM(
            embed_size=64, hidden_size=128, encode_size=126, obs_encode_size=128)
        self.lane_aggregation = AttentionalAggregation(input_encoding_size=128, output_size=256)
        self.prediction_layer = DistributionalScoring(
            obs_enc_size=128, lane_enc_size=192, aggr_enc_size=512, mlp_size=[33, 1])

    def forward(self, X):
        '''
            - obs_hist_size: N x 1
            - obs_features: N x 180
            - backward_lane_points: M x 20
            - lane_features: M x 600
            - is_self_lane: M x 1
            - same_obs_mask: M x 1
        '''
        obs_hist_size, obs_features, backward_lane_points, lane_features, is_self_lane, same_obs_mask = X
        obs_hist_size = obs_hist_size.float()
        obs_features = obs_features.float()
        lane_features = lane_features.float()
        is_self_lane = is_self_lane.float()
        same_obs_mask = same_obs_mask.float()
        N = obs_features.size(0)
        M = lane_features.size(0)

        # N x 128
        obs_enc = self.vehicle_encoding(obs_features, obs_hist_size)

        # M x 64
        backward_enc = self.backward_lane_encoding(
            backward_lane_points, obs_hist_size, same_obs_mask)

        lane_enc_1 = self.lane_encoding(lane_features, obs_enc, same_obs_mask)
        lane_enc_2 = lane_features[:, 0].view(M, 1)
        lane_enc_3 = is_self_lane.view(M, 1)
        # M x 128
        lane_enc = torch.cat((lane_enc_1, lane_enc_2, lane_enc_3), 1)

        # N x 512
        aggr_enc = self.lane_aggregation(obs_enc, lane_enc, same_obs_mask)

        # M x 192
        total_lane_enc = torch.cat((lane_enc, backward_enc), 1)

        out = self.prediction_layer(obs_enc, total_lane_enc, aggr_enc, same_obs_mask)

        return out


class VehicleLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(VehicleLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(8, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.vehicle_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1,
                                   batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size * 8, encode_size),
            nn.ReLU(),
        )

    def forward(self, obs_features):
        '''Forward function
            - obs_features: N x 180
            - hist_size: N x 1

            output: N x encode_size
        '''
        N = obs_features.size(0)

        # Input embedding.
        # (N x 20 x 1)
        obs_x = obs_features[:, 1::9].view(N, 20, 1)
        obs_y = obs_features[:, 2::9].view(N, 20, 1)
        vel_x = obs_features[:, 3::9].view(N, 20, 1)
        vel_y = obs_features[:, 4::9].view(N, 20, 1)
        acc_x = obs_features[:, 5::9].view(N, 20, 1)
        acc_y = obs_features[:, 6::9].view(N, 20, 1)
        vel_heading = obs_features[:, 7::9].view(N, 20, 1)
        vel_heading_changing_rate = obs_features[:, 8::9].view(N, 20, 1)
        # (N x 20 x 8)
        obs_position = torch.cat((obs_x, obs_y, vel_x, vel_y, acc_x, acc_y,
                                  vel_heading, vel_heading_changing_rate), 2).float()
        # (N x 20 x embed_size)
        obs_embed = self.embed(obs_position.view(N * 20, 8)).view(N, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        # (N x 20 x 2*hidden_size)
        obs_states, _ = self.vehicle_rnn(obs_embed, (h0, c0))
        # (N x 2*hidden_size)
        front_states = obs_states[:, 0, :]
        back_states = obs_states[:, -1, :]
        max_states, _ = torch.max(obs_states, 1)
        avg_states = torch.mean(obs_states, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, avg_states), 1)
        out = self.encode(out)
        return out


class VehicleDynamicLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(VehicleDynamicLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(8, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.vehicle_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size * 3, encode_size),
            nn.ReLU(),
        )

    def forward(self, obs_features, hist_size):
        '''Forward function
            - obs_features: N x 180
            - hist_size: N x 1

            output: N x encode_size
        '''
        N = obs_features.size(0)

        # Input preprocessing.
        # (N x 20 x 1)
        obs_x = obs_features[:, 1::9].view(N, 20, 1)
        obs_y = obs_features[:, 2::9].view(N, 20, 1)
        vel_x = obs_features[:, 3::9].view(N, 20, 1)
        vel_y = obs_features[:, 4::9].view(N, 20, 1)
        acc_x = obs_features[:, 5::9].view(N, 20, 1)
        acc_y = obs_features[:, 6::9].view(N, 20, 1)
        vel_heading = obs_features[:, 7::9].view(N, 20, 1)
        vel_heading_changing_rate = obs_features[:, 8::9].view(N, 20, 1)
        # (N x 20 x 8)
        obs_fea = torch.cat((obs_x, obs_y, vel_x, vel_y, acc_x, acc_y,
                             vel_heading, vel_heading_changing_rate), 2).float()

        # Sort based on dscending sequence lengths.
        # original_idx = torch.arange(N)                                # [0, 1, 2, 3, 4]
        seq_lengths, sorted_idx = hist_size.sort(0, descending=True)    # [1, 3, 2, 0, 4]
        _, recover_idx = sorted_idx.sort(0, descending=False)           # [3, 0, 2, 1, 4]
        seq_lengths = seq_lengths.view(-1).long()
        sorted_idx = sorted_idx.view(-1).long()
        recover_idx = recover_idx.view(-1).long()
        obs_fea = obs_fea.index_select(0, sorted_idx)

        # Input embedding.
        # (N x 20 x embed_size)
        obs_embed = self.embed(obs_fea.view(N * 20, 8)).view(N, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        packed_input = pack_padded_sequence(obs_embed, seq_lengths.cpu(), batch_first=True)
        packed_output, (ht, ct) = self.vehicle_rnn(packed_input)
        # (N x 20 x hidden_size)
        obs_states, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        min_val = torch.min(obs_states)
        obs_states_for_maxpool, _ = pad_packed_sequence(packed_output, batch_first=True,
                                                        padding_value=min_val.item() - 0.1)
        # (N x hidden_size)
        final_states = ht.view(N, self.hidden_size)
        max_states, _ = torch.max(obs_states_for_maxpool, 1)
        avg_states = torch.sum(obs_states, 1) / seq_lengths.float().view(N, 1)

        # Encoding
        out = torch.cat((final_states, max_states, avg_states), 1)
        out = self.encode(out)

        # Recover the original sequence.
        out = out.index_select(0, recover_idx)

        return out


class BackwardLaneLSTM(nn.Module):
    def __init__(self, embed_size=16, hidden_size=64, encode_size=64):
        super(BackwardLaneLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(1, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.vehicle_rnn = nn.LSTM(embed_size, hidden_size, num_layers=1,
                                   batch_first=True, bidirectional=False)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size * 3, encode_size),
            nn.ReLU(),
        )

    def forward(self, obs_backward_features, hist_size, same_obs_mask):
        '''Forward function
            - obs_features: M x 20
            - hist_size: N x 1

            output: M x encode_size
        '''
        M = obs_backward_features.size(0)
        # N = hist_size.size(0)

        # Input embedding.
        obs_fea = obs_backward_features.float()

        # Sort the obstacle_features by descending history-length.
        new_hist_size = cuda(torch.zeros(M, 1))
        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_N = torch.sum(curr_mask).item()
            new_hist_size[curr_mask, :] = hist_size[obs_id, 0].view(1, 1).repeat(curr_N, 1)
        # original_idx = torch.arange(M)                                    # [0, 1, 2, 3, 4]
        seq_lengths, sorted_idx = new_hist_size.sort(0, descending=True)    # [1, 3, 2, 0, 4]
        _, recover_idx = sorted_idx.sort(0, descending=False)               # [3, 0, 2, 1, 4]
        seq_lengths = seq_lengths.view(-1).long()
        sorted_idx = sorted_idx.view(-1).long()
        recover_idx = recover_idx.view(-1).long()
        obs_fea = obs_fea.index_select(0, sorted_idx)

        # (M x 20 x embed_size)
        obs_embed = self.embed(obs_fea.view(M * 20, 1))
        obs_embed = obs_embed.view(M, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, M, 1), self.c0.repeat(1, M, 1)
        packed_input = pack_padded_sequence(obs_embed, seq_lengths.cpu(), batch_first=True)
        packed_output, (ht, ct) = self.vehicle_rnn(packed_input)
        # (M x 20 x hidden_size)
        obs_states, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        min_val = torch.min(obs_states)
        obs_states_for_maxpool, _ = pad_packed_sequence(packed_output, batch_first=True,
                                                        padding_value=min_val.item() - 0.1)
        # (M x hidden_size)
        final_states = ht.view(M, self.hidden_size)
        max_states, _ = torch.max(obs_states_for_maxpool, 1)
        avg_states = torch.sum(obs_states, 1) / seq_lengths.float().view(M, 1)

        # Encoding
        out = torch.cat((final_states, max_states, avg_states), 1)
        out = self.encode(out)

        # Recover the original sequence.
        out = out.index_select(0, recover_idx)

        return out


class LaneLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128):
        super(LaneLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.single_lane_rnn = nn.LSTM(embed_size, hidden_size,
                                       num_layers=1, batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size * 8, encode_size),
            nn.ReLU(),
        )

    def forward(self, lane_features):
        '''Forward function:
            - lane_features: N x 400

            output: N x encode_size
        '''
        forward_lane_features = lane_features[:, -400:].contiguous()
        N = forward_lane_features.size(0)

        # Input embedding.
        # (N x 100 x embed_size)
        lane_input = forward_lane_features.float().view(N, 100, 4)
        lane_input = lane_input[:, ::5, :]
        lane_input = lane_input.view(N * 20, 4)
        lane_embed = self.embed(lane_input).view(N, 20, self.embed_size)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, N, 1), self.c0.repeat(1, N, 1)
        # (N x 20 x 2*hidden_size)
        lane_states, _ = self.single_lane_rnn(lane_embed, (h0, c0))
        # (N x 2*hidden_size)
        front_states = lane_states[:, 0, :]
        back_states = lane_states[:, -1, :]
        max_states, _ = torch.max(lane_states, 1)
        avg_states = torch.mean(lane_states, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, avg_states), 1)
        out = self.encode(out)
        return out


class AttentionalLaneLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden_size=128, encode_size=128, obs_encode_size=128):
        super(AttentionalLaneLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encode_size = encode_size
        self.obs_encode_size = obs_encode_size

        self.embed = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        self.obs_attention = torch.nn.Sequential(
            nn.Linear(obs_encode_size, 100),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

        h0 = torch.zeros(2, 1, hidden_size)
        c0 = torch.zeros(2, 1, hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        self.single_lane_rnn = nn.LSTM(embed_size, hidden_size,
                                       num_layers=1, batch_first=True, bidirectional=True)

        self.encode = torch.nn.Sequential(
            nn.Linear(hidden_size * 8, encode_size),
            nn.ReLU(),
        )

    def forward(self, lane_features, obs_encoding, same_obs_mask):
        '''Forward function:
            - lane_features: M x 600
            - obs_encoding: N x 128
            - same_obs_mask: M x 1

            output: M x encode_size
        '''
        M = lane_features.size(0)
        N = obs_encoding.size(0)

        # Input embedding.
        # (M x 150 x embed_size)
        forward_lane_features = lane_features[:, -400:].contiguous()
        lane_embed = self.embed(forward_lane_features.float().view(M * 100, 4)
                                ).view(M, 100, self.embed_size)

        # obs_feature attention.
        # (N x 100)
        attention_scores = self.obs_attention(obs_encoding)
        new_attention_scores = cuda(torch.zeros(M, 100))
        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_N = torch.sum(curr_mask).item()
            new_attention_scores[curr_mask, :] = attention_scores[obs_id, :].view(
                1, 100).repeat(curr_N, 1)
        # (M x 100 x hidden_size)
        attention_scores = new_attention_scores.view(M, 100, 1).repeat(1, 1, self.hidden_size * 2)

        # Run through RNN.
        h0, c0 = self.h0.repeat(1, M, 1), self.c0.repeat(1, M, 1)
        # (M x 100 x hidden_size)
        lane_states, _ = self.single_lane_rnn(lane_embed, (h0, c0))
        # (M x hidden_size)
        front_states = lane_states[:, 0, :]
        back_states = lane_states[:, -1, :]
        max_states, _ = torch.max(lane_states, 1)
        attentiona_states = torch.sum(lane_states * attention_scores, 1)

        # Encoding
        out = torch.cat((front_states, back_states, max_states, attentiona_states), 1)
        out = self.encode(out)
        return out


class SimpleAggregation(nn.Module):
    def __init__(self):
        super(SimpleAggregation, self).__init__()

    def forward(self, obs_encoding, lane_encoding, same_obs_mask):
        '''Forward function
            - obs_encoding: N x input_encoding_size
            - lane_encoding: M x input_encoding_size
            - same_obs_mask: M x 1

            output: N x output_size
        '''
        N = obs_encoding.size(0)
        input_encoding_size = obs_encoding.size(1)
        out = cuda(torch.zeros(N, input_encoding_size * 2))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()

            # (curr_num_lane x input_encoding_size)
            curr_lane_encoding = lane_encoding[curr_mask, :].view(curr_num_lane, -1)
            # (1 x input_encoding_size)
            curr_lane_maxpool, _ = torch.max(curr_lane_encoding, 0, keepdim=True)
            # (1 x input_encoding_size)
            curr_lane_avgpool = torch.mean(curr_lane_encoding, 0, keepdim=True)
            out[obs_id, :] = torch.cat((curr_lane_maxpool, curr_lane_avgpool), 1)

        return out


# TODO(jiacheng):
#   - Add pairwise attention between obs_encoding and every lane_encoding during aggregating.
class AttentionalAggregation(nn.Module):
    def __init__(self, input_encoding_size=128, output_size=512):
        super(AttentionalAggregation, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.output_size = output_size

        self.encode = torch.nn.Sequential(
            nn.Linear(input_encoding_size, output_size),
            nn.ReLU(),
        )

    def forward(self, obs_encoding, lane_encoding, same_obs_mask):
        '''Forward function
            - obs_encoding: N x input_encoding_size
            - lane_encoding: M x input_encoding_size
            - same_obs_mask: M x 1

            output: N x output_size
        '''
        N = obs_encoding.size(0)
        out = cuda(torch.zeros(N, self.output_size * 2))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()

            # (curr_num_lane x input_encoding_size)
            curr_lane_encoding = lane_encoding[curr_mask, :].view(curr_num_lane, -1)
            curr_lane_encoding = self.encode(curr_lane_encoding)
            # (1 x input_encoding_size)
            curr_lane_maxpool, _ = torch.max(curr_lane_encoding, 0, keepdim=True)
            # (1 x input_encoding_size)
            curr_lane_avgpool = torch.mean(curr_lane_encoding, 0, keepdim=True)
            out[obs_id, :] = torch.cat((curr_lane_maxpool, curr_lane_avgpool), 1)

        return out


class DistributionalScoring(nn.Module):
    def __init__(self, obs_enc_size=128, lane_enc_size=128,
                 aggr_enc_size=1024, mlp_size=[600, 100, 16, 1]):
        super(DistributionalScoring, self).__init__()
        self.mlp = generate_mlp(
            [obs_enc_size + lane_enc_size + aggr_enc_size] + mlp_size, dropout=0.0, last_layer_nonlinear=False)
        self.softmax_layer = nn.Softmax()

    def forward(self, obs_encoding, lane_encoding, aggregated_info, same_obs_mask):
        '''Forward function (M >= N)
            - obs_encoding: N x obs_enc_size
            - lane_encoding: M x lane_enc_size
            - aggregated_info: N x aggr_enc_size
        '''
        N = obs_encoding.size(0)
        M = lane_encoding.size(0)
        out = cuda(torch.zeros(M, 1))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()

            curr_obs_enc = obs_encoding[obs_id, :].view(1, -1)
            curr_obs_enc = curr_obs_enc.repeat(curr_num_lane, 1)
            curr_agg_info = aggregated_info[obs_id, :].view(1, -1)
            curr_agg_info = curr_agg_info.repeat(curr_num_lane, 1)
            curr_lane_enc = lane_encoding[curr_mask, :].view(curr_num_lane, -1)

            curr_encodings = torch.cat((curr_obs_enc, curr_agg_info, curr_lane_enc), 1)
            curr_scores = self.mlp(curr_encodings).view(-1)
            curr_scores = self.softmax_layer(curr_scores)
            out[curr_mask, :] = curr_scores.view(curr_num_lane, 1)

        return out


class ClassificationLoss:
    def loss_fn(self, y_pred, y_true_tuple):
        y_labels, y_is_cutin, y_same_obs_mask = y_true_tuple
        y_pred_score = -torch.log(y_pred).float()
        y_labels = y_labels.float()

        total_num_data_pt = y_same_obs_mask.max().long().item() + 1
        scores = cuda(torch.zeros(total_num_data_pt))

        for obs_id in range(total_num_data_pt):
            curr_mask = (y_same_obs_mask[:, 0] == obs_id)
            curr_pred = y_pred_score[curr_mask, 0]
            curr_true = y_labels[curr_mask, 0]
            curr_score = torch.sum(curr_pred * curr_true)
            scores[obs_id] = curr_score

        final_loss = torch.mean(scores)
        return final_loss

    def loss_info(self, y_pred, y_true_tuple):
        y_labels, y_is_cutin, y_same_obs_mask = y_true_tuple
        total_num_data_pt = 0.0
        total_correct_data_pt = 0.0

        for obs_id in range(y_same_obs_mask.max().long().item() + 1):
            curr_mask = (y_same_obs_mask[:, 0] == obs_id)

            curr_pred = y_pred[curr_mask, 0]
            curr_true = y_labels[curr_mask, 0]
            if curr_true[torch.argmax(curr_pred)] == 1:
                total_correct_data_pt += 1
            total_num_data_pt += 1

        return total_correct_data_pt / total_num_data_pt

    def loss_debug(self, y_pred, y_true_tuple):
        y_labels, y_is_cutin, y_same_obs_mask = y_true_tuple
        wrong_pred_ids = []

        for obs_id in range(y_same_obs_mask.max().long().item() + 1):
            curr_mask = (y_same_obs_mask[:, 0] == obs_id)

            curr_pred = y_pred[curr_mask, 0]
            curr_true = y_labels[curr_mask, 0]
            if curr_true[torch.argmax(curr_pred)] != 1:
                wrong_pred_ids.append(obs_id)

        return wrong_pred_ids
