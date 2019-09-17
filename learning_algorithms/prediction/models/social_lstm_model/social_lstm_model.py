#!/usr/bin/env python

import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


'''
========================================================================
Dataset set-up
========================================================================
'''

'''
Read files that contain training data line by line.
Within each line, use "delim" as the deliminator.
'''


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class HumanTrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 min_ped=0, delim='\t'):
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(data_dir, _path) for _path in all_files]
        seq_len = obs_len + pred_len
        num_peds_in_scene = []
        scene_list = []
        scene_rel_list = []
        scene_timestamp_mask = []
        scene_is_predictable_list = []

        # Go through all the files that contain data
        for path in all_files:
            data = read_file(path, delim)

            # Organize the data in the following way:
            # All obstacles belonging to the same timestamp are clustered
            # together; timestamp is sorted.
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[data[:, 0] == frame, :])
            num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

            # A scene is defined as a stream of frames lasting seq_len frames.
            # During training, the interactions among all obstacles in the
            # same scene are considered.

            # Go through every scene:
            for idx in range(0, num_sequences * skip + 1, skip):
                curr_scene_data = np.concatenate(
                    frame_data[idx:idx + seq_len], axis=0)
                peds_in_curr_scene_data = np.unique(curr_scene_data[:, 1])
                curr_scene = np.zeros(
                    (len(peds_in_curr_scene_data), 2, seq_len))
                curr_scene_rel = np.zeros(
                    (len(peds_in_curr_scene_data), 2, seq_len))
                curr_scene_timestamp_mask = np.zeros(
                    (len(peds_in_curr_scene_data), seq_len))
                curr_scene_is_predictable = np.zeros(
                    (len(peds_in_curr_scene_data), 1))
                # Go through every obstacle in the current scene:
                num_peds_considered = 0
                for i, ped_id in enumerate(peds_in_curr_scene_data):
                    curr_ped = curr_scene_data[curr_scene_data[:, 1] == ped_id, :]
                    curr_ped = np.around(curr_ped, decimals=4)
                    time_begin = frames.index(curr_ped[0, 0]) - idx
                    time_end = frames.index(curr_ped[-1, 0]) - idx + 1
                    # If this obstacle doesn't have enough number of frames,
                    # mark it as non-predictable. Vice versa.
                    curr_ped_is_predictable = False
                    if time_end == seq_len and time_begin < 2:
                        curr_ped_is_predictable = True
                    # Get the coordinates of positions and make them relative.
                    curr_ped = np.transpose(curr_ped[:, 2:])
                    curr_ped_rel = np.zeros(curr_ped.shape)
                    curr_ped_timestamp_mask = np.ones((1, time_end - time_begin))
                    curr_ped_rel[:, 1:] = curr_ped[:, 1:] - curr_ped[:, :-1]
                    # Update into curr_scene matrix.
                    curr_scene[i, :, time_begin:time_end] = curr_ped
                    curr_scene_rel[i, :, time_begin:time_end] = curr_ped_rel
                    curr_scene_timestamp_mask[i, time_begin:time_end] = \
                        curr_ped_timestamp_mask
                    curr_scene_is_predictable[i] = curr_ped_is_predictable
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_scene.append(num_peds_considered)
                    scene_list.append(np.transpose(curr_scene, (0, 2, 1)))
                    scene_rel_list.append(np.transpose(curr_scene_rel, (0, 2, 1)))
                    scene_timestamp_mask.append(curr_scene_timestamp_mask)
                    scene_is_predictable_list.append(curr_scene_is_predictable)

        self.num_scene = len(scene_list)
        scene_list = np.concatenate(scene_list, axis=0)
        scene_rel_list = np.concatenate(scene_rel_list, axis=0)
        scene_timestamp_mask = np.concatenate(scene_timestamp_mask, axis=0)
        scene_is_predictable_list = np.concatenate(
            scene_is_predictable_list, axis=0)

        start_idx = [0] + np.cumsum(num_peds_in_scene).tolist()
        self.scene_start_end_idx = [
            (start, end) for start, end in zip(start_idx[:-1], start_idx[1:])]
        self.past_traj = scene_list[:, :obs_len, :]
        self.pred_traj = scene_list[:, obs_len:, :]
        self.past_traj_rel = scene_rel_list[:, :obs_len, :]
        self.pred_traj_rel = scene_rel_list[:, obs_len:, :]
        self.past_traj_timestamp_mask = scene_timestamp_mask[:, :obs_len]
        self.is_predictable = scene_is_predictable_list

    def __len__(self):
        return self.num_scene

    def __getitem__(self, idx):
        start, end = self.scene_start_end_idx[idx]
        out = (self.past_traj[start:end], self.past_traj_rel[start:end],
               self.pred_traj[start:end], self.pred_traj_rel[start:end],
               self.past_traj_timestamp_mask[start:end],
               self.is_predictable[start:end])
        return out


def collate_scenes(batch):
    # batch is a list of tuples
    # unzip to form list of np-arrays
    past_traj, past_traj_rel, pred_traj, pred_traj_rel, \
        past_traj_timestamp_mask, is_predictable = zip(*batch)
    same_scene_mask = [element.shape[0] for element in past_traj]
    past_traj = np.concatenate(past_traj)
    past_traj_rel = np.concatenate(past_traj_rel)
    pred_traj = np.concatenate(pred_traj)
    pred_traj_rel = np.concatenate(pred_traj_rel)
    past_traj_timestamp_mask = np.concatenate(past_traj_timestamp_mask)
    is_predictable = np.concatenate(is_predictable)

    temp_mask = []
    for i, length in enumerate(same_scene_mask):
        temp_mask.append(np.ones((length, 1)) * i)
    same_scene_mask = \
        [np.ones((length, 1))*i for i, length in enumerate(same_scene_mask)]
    same_scene_mask = np.concatenate(same_scene_mask)

    return (torch.from_numpy(past_traj), torch.from_numpy(past_traj_rel),
            torch.from_numpy(past_traj_timestamp_mask),
            torch.from_numpy(is_predictable), torch.from_numpy(same_scene_mask)),\
        (torch.from_numpy(pred_traj), torch.from_numpy(pred_traj_rel),
         torch.from_numpy(is_predictable))


'''
========================================================================
Model definition
========================================================================
'''


class SocialPooling(nn.Module):
    def __init__(self, grid_size=2, area_span=1.6):
        super(SocialPooling, self).__init__()
        self.grid_size = grid_size
        self.area_span = area_span

    def decide_grid(self, curr_pos_t):
        N = curr_pos_t.size(0)

        # N x N x 2
        rel_pos_matrix = torch.transpose(curr_pos_t.repeat(1, N, 1), 0, 1) -\
            curr_pos_t.repeat(1, N, 1)
        # N x N
        eps = 1e-2
        mask_within_pooling_area = \
            (rel_pos_matrix[:, :, 0] < self.area_span / 2.0-eps) * \
            (rel_pos_matrix[:, :, 0] > -self.area_span / 2.0+eps) * \
            (rel_pos_matrix[:, :, 1] < self.area_span / 2.0-eps) * \
            (rel_pos_matrix[:, :, 1] > -self.area_span / 2.0+eps)
        mask_within_pooling_area = mask_within_pooling_area.float()
        mask_within_pooling_area -= torch.eye(N).cuda()

        # N x N
        mask_grid_id = torch.floor(
            (rel_pos_matrix.float() + torch.tensor(self.area_span / 2.0)) /
            torch.tensor(self.area_span / self.grid_size))
        mask_grid_id = mask_grid_id[:, :, 0] * self.grid_size + \
            mask_grid_id[:, :, 1]
        mask_grid_id *= mask_within_pooling_area
        mask_grid_id = mask_grid_id.long()

        return mask_within_pooling_area, mask_grid_id

    def forward(self, ht, pos_t, same_scene_mask):
        hidden_size = ht.size(2)
        ht_pooled = torch.zeros(ht.size(0), self.grid_size ** 2, hidden_size).cuda()

        all_scene_ids = torch.unique(same_scene_mask).cpu().numpy().tolist()
        #print (all_scene_ids)
        N_filled = 0
        for scene_id in all_scene_ids:
            # N x 1 x hidden_size
            #print (ht.shape)
            curr_ht = ht[same_scene_mask[:, 0] == scene_id, :, :]
            #print (curr_ht.shape)
            # N x 1 x 2
            curr_pos_t = pos_t[same_scene_mask[:, 0] == scene_id, :, :]
            curr_N = curr_ht.size(0)
            if (curr_N == 0):
                continue
            curr_ht = curr_ht.view(curr_N, 1, -1)

            mask_within_pooling_area, mask_grid_id = self.decide_grid(curr_pos_t)
            #print (mask_within_pooling_area)
            #print (mask_grid_id)
            # N x N x hidden_size
            ht_matrix = torch.transpose(curr_ht.repeat(1, curr_N, 1), 0, 1).float()
            ht_matrix *= mask_within_pooling_area.\
                reshape((curr_N, curr_N, 1)).repeat(1, 1, hidden_size)

            # N x grid_size ^ 2 x hidden_size
            mask_grid_id = mask_grid_id.\
                reshape((curr_N, curr_N, 1)).repeat(1, 1, hidden_size)
            curr_ht_pooled = torch.zeros(curr_N, self.grid_size ** 2, hidden_size).cuda()
            #print ('=================')
            #print (mask_grid_id)
            #print (mask_grid_id.max())
            #print (mask_grid_id.min())
            #print (mask_grid_id.shape)
            #print (curr_ht_pooled.shape)
            #print (ht_matrix.shape)
            curr_ht_pooled = curr_ht_pooled.scatter_add(1, mask_grid_id, ht_matrix)

            ht_pooled[N_filled:N_filled+curr_N, :, :] = curr_ht_pooled
            N_filled += curr_N

        return ht_pooled


class SocialLSTM(nn.Module):
    def __init__(self, pred_len=12, grid_size=2, area_span=2.0, embed_size=64,
                 hidden_size=128):
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

        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers=1,
                            batch_first=True)
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
        # Get needed input data
        traj, traj_rel, time_mask, is_predictable_mask, same_scene_mask = X

        # Get dimensions
        N = traj.size(0)
        observation_len = traj.size(1)

        # Look at past traj history
        # N x 1 x hidden_size
        ht, ct = self.h0.repeat(N, 1, 1), self.c0.repeat(N, 1, 1)
        for t in range(observation_len):
            # Filter out those points that doesn't exist at the current timestamp
            curr_mask = (time_mask[:, t] == 1)
            curr_N = torch.sum(curr_mask).item()

            # curr_N x 1 x 2
            curr_point = traj[curr_mask, t, :].reshape(curr_N, 1, 2)
            curr_point_rel = traj_rel[curr_mask, t, :].reshape(curr_N, 1, 2)
            # curr_N x 1
            curr_same_scene_mask = same_scene_mask[curr_mask]
            # curr_N x 1 x hidden_size
            curr_ht, curr_ct = ht[curr_mask, :, :], ct[curr_mask, :, :]

            # Apply social-pooling
            # N x (grid_size ** 2) x hidden_size
            Ht = self.social_pooling(
                curr_ht, curr_point, curr_same_scene_mask)

            # Apply embeddings
            # N x embed_size
            et = self.pos_embedding(curr_point_rel.view(curr_N, 2).float())
            at = self.social_embedding(Ht.view(curr_N, -1))

            # Step through RNN
            _, (curr_ht, curr_ct) = self.lstm(
                torch.cat((et, at), 1).view(curr_N, 1, -1),
                (curr_ht.view(1, curr_N, -1), curr_ct.view(1, curr_N, -1)))
            ht[curr_mask, :, :], ct[curr_mask, :, :] = \
                curr_ht.view(curr_N, 1, -1), curr_ct.view(curr_N, 1, -1)

        # Predict future traj
        pred_mask = (time_mask[:, -1] == 1)
        pred_N = torch.sum(pred_mask).item()
        if(pred_N == 0):
            return None
        pred_same_scene_mask = same_scene_mask[pred_mask]
        pred_point = traj[pred_mask, -1, :].float().reshape(pred_N, 1, 2)
        pred_ht, pred_ct = ht[pred_mask, :, :], ct[pred_mask, :, :]
        # (ux, uy, sigma_x, sigma_y, rho)
        pred_out = torch.zeros(pred_N, self.pred_len, 5).cuda()
        for t in range(self.pred_len):
            pred_out[:, t, :] = \
                self.pred_layer(pred_ht.view(pred_N, -1)).view(pred_N, 5).float()
            pred_point_rel = pred_out[:, t, :2].float().view(pred_N, 1, 2).cuda()

            pred_point = pred_point + pred_point_rel
            Ht = self.social_pooling(
                pred_ht, pred_point, pred_same_scene_mask)
            pred_point_rel = pred_point_rel.view(pred_N, 2).clone()
            et = self.pos_embedding(pred_point_rel)
            at = self.social_embedding(Ht.view(pred_N, -1))
            _, (pred_ht, pred_ct) = self.lstm(
                torch.cat((et, at), 1).view(pred_N, 1, -1),
                (pred_ht.view(1, pred_N, -1), pred_ct.view(1, pred_N, -1)))
            pred_ht = pred_ht.view(pred_N, 1, -1)
        pred_out_all = torch.zeros(N, self.pred_len, 5).cuda()
        pred_out_all[pred_mask, :, :] = pred_out
        return pred_out_all[is_predictable_mask[:, 0] == 1, :, :]


class ProbablisticTrajectoryLoss:
    def loss_fn(self, y_pred, y_true):
        if y_pred is None:
            return 0
        # y_pred: N x pred_len x 5
        # y_true: (pred_traj, pred_traj_rel)  N x pred_len x 2
        mux, muy, sigma_x, sigma_y, corr = y_pred[:, :, 0], y_pred[:, :, 1],\
            y_pred[:, :, 2], y_pred[:, :, 3], y_pred[:, :, 4]
        is_predictable = y_true[2].long()
        x, y = y_true[1][is_predictable[:, 0] == 1, :, 0].float(), \
            y_true[1][is_predictable[:, 0] == 1, :, 1].float()
        N = y_pred.size(0)
        if N == 0:
            return 0

        eps = 1e-10

        z = ((x-mux)/(eps+sigma_x))**2 + ((y-muy)/(eps+sigma_y))**2 - \
            2*corr*(x-mux)*(y-muy)/(sigma_x*sigma_y+eps)
        #print (z)
        P = 1/(2*np.pi*sigma_x*sigma_y*torch.sqrt(1-corr**2)+eps) * \
            torch.exp(-z/(2*(1-corr**2)))

        loss = torch.clamp(P, min=eps)
        #print (loss)
        loss = -loss.log()
        return torch.sum(loss)/N

    def loss_valid(self, y_pred, y_true):
        loss = nn.MSELoss()

        is_predictable = y_true[2].long()
        out = loss(y_pred[:, :, :2], y_true[1][is_predictable[:, 0] == 1, :, :].float())
        return out

    def loss_info(self, y_pred, y_true):
        is_predictable = y_true[2].long()
        out = y_pred[:, :, :2] - y_true[1][is_predictable[:, 0] == 1, :, :].float()
        out = out ** 2
        out = torch.sum(out, 2)
        out = torch.sqrt(out)
        out = torch.mean(out)
        return out
