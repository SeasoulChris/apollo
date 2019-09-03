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
from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *

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
            key = os.path.basename(img_name).replace(".png", "")
            pos_dict = np.load(os.path.join(os.path.dirname(img_name), 'obs_pos.npy')).item()
            past_pos = pos_dict[key]
            label_dict = np.load(os.path.join(os.path.dirname(img_name).replace(
                "image-feature", "features-san-mateo-new").replace("image-valid", "features-san-mateo-new"), 'future_status.npy')).item()
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
    def __init__(self, num_pred_points, num_history_points,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapModel, self).__init__()

        self.cnn = cnn_net(pretrained=pretrained)
        self.num_pred_points = num_pred_points
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features + num_history_points * 2, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, num_pred_points * 2)
        )

    def forward(self, X):
        img, obs_pos, _, _ = X
        out = self.cnn(img)
        out = out.view(out.size(0), -1)
        obs_pos = obs_pos.view(obs_pos.size(0), -1)
        out = torch.cat([out, obs_pos], -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.num_pred_points, 2)
        return out


class SemanticMapLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        out = y_pred - y_true
        out = torch.mean(out ** 2)
        return out


class SemanticMapSelfLSTMModel(nn.Module):
    def __init__(self, pred_len, num_history_points,
                 embed_size=64, hidden_size=128,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapSelfLSTMModel, self).__init__()
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = self.cnn.fc.in_features
        self.pred_len = pred_len
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.disp_embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=1, batch_first=True)

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 2),
        )

    def forward(self, X):
        img, obs_pos, obs_hist_size, obs_pos_rel, _, _, _, _ = X
        N = obs_pos.size(0)
        observation_len = obs_pos.size(1)
        ht, ct = self.h0.repeat(N, 1), self.h0.repeat(N, 1)

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        pred_mask = cuda(torch.ones(N))
        pred_out = cuda(torch.zeros(N, self.pred_len, 2))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))

        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                ts_obs_mask = (obs_hist_size > observation_len-t).long().view(-1)
                curr_obs_pos_rel = obs_pos_rel[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                ts_obs_mask = pred_mask
                pred_input = torch.cat((ht.clone(), img_embedding), 1)
                pred_out[:, t-observation_len, :] = self.pred_layer(pred_input).float().clone()
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

        return pred_traj


class SemanticMapSocialAttentionModel(nn.Module):
    '''
    Semantic map model with social attention
    '''
    def __init__(self, pred_len, num_history_points,
                 embed_size=64, edge_hidden_size=256, node_hidden_size=128, attention_dim=64,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapSocialAttentionModel, self).__init__()
        self.pred_len = pred_len

        # CNN
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = self.cnn.fc.in_features
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.edge_hidden_size = edge_hidden_size

        # Target hidden state
        target_h0 = torch.zeros(1, node_hidden_size)
        target_c0 = torch.zeros(1, node_hidden_size)
        nn.init.xavier_normal_(target_h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(target_c0, gain=nn.init.calculate_gain('relu'))
        self.target_h0 = nn.Parameter(target_h0, requires_grad=True)
        self.target_c0 = nn.Parameter(target_c0, requires_grad=True)

        # Target RNN flow
        self.target_rel_disp_embedding = nn.Linear(2, embed_size)
        self.target_lstm = nn.LSTM(embed_size, node_hidden_size,
                                   num_layers=1, batch_first=True)
        self.target_attention_embedding = nn.Linear(2, attention_dim)

        # Nearby hidden state
        nearby_h0 = torch.zeros(1, 1, edge_hidden_size)
        nearby_c0 = torch.zeros(1, 1, edge_hidden_size)
        nn.init.xavier_normal_(nearby_h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(nearby_c0, gain=nn.init.calculate_gain('relu'))
        self.nearby_h0 = nn.Parameter(nearby_h0, requires_grad=True)
        self.nearby_c0 = nn.Parameter(nearby_c0, requires_grad=True)

        # Nearby RNN flow
        self.nearby_rel_disp_embedding = nn.Linear(2, embed_size)
        self.nearby_lstm = nn.LSTM(embed_size, edge_hidden_size, num_layers=1, batch_first=True)
        self.nearby_attention_embedding = nn.Linear(2, attention_dim)

        self.attention_dim = attention_dim

        # Prediction FC layer
        self.pred_layer = nn.Linear(node_hidden_size + self.cnn_out_size + edge_hidden_size, 2)

    def forward(self, X):
        img, target_obs_pos_rel, target_obs_hist_size, all_obs_pos_rel, \
             target_obs_pos, nearby_obs_pos, nearby_obs_hist_sizes, nearby_obs_pos_rel = X

        observation_len = target_obs_pos.size(1)
        num_nearby_obs = nearby_obs_pos.size(1)
        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        # Initialize target RNN
        target_ht = self.target_h0
        target_ct = self.target_c0
        # Initialize nearby RNN
        if num_nearby_obs >= 1:
            nearby_ht_list = self.nearby_h0.repeat(num_nearby_obs, 1, 1)
            nearby_ct_list = self.nearby_c0.repeat(num_nearby_obs, 1, 1)
        else:
            nearby_ht_list = self.nearby_h0
            nearby_ct_list = self.nearby_c0

        pred_mask = cuda(torch.ones(1))
        pred_out = cuda(torch.zeros(1, self.pred_len, 2))
        pred_traj = cuda(torch.zeros(1, self.pred_len, 2))

        Ht = self.nearby_h0

        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                target_ts_obs_mask = target_obs_hist_size > observation_len - t
                curr_target_obs_pos = target_obs_pos[0, t, :]
                curr_target_obs_pos_rel = target_obs_pos_rel[0, t, :]
                nearby_ts_obs_mask = (nearby_obs_hist_sizes > observation_len-t).view(-1)
                curr_nearby_obs_pos = nearby_obs_pos[:, :, t, :].float()
                curr_nearby_obs_pos_rel = nearby_obs_pos_rel[:, :, t, :].float()
            else:
                target_ts_obs_mask = pred_mask
                nearby_ts_obs_mask = (nearby_obs_hist_sizes > -1).view(-1)
                pred_input = torch.cat((target_ht.clone(), img_embedding, Ht.view(1, -1)), 1)
                pred_out[:, t-observation_len, :] = self.pred_layer(pred_input).float().clone()
                curr_target_obs_pos_rel = pred_out[:, t-observation_len, :2]
                curr_target_obs_pos = curr_target_obs_pos + curr_target_obs_pos_rel
                pred_traj[:, t-observation_len, :] = curr_target_obs_pos.clone()

            # Target obstacles forward
            if target_ts_obs_mask:
                target_disp_embedding = self.target_rel_disp_embedding(
                    curr_target_obs_pos_rel.clone()).view(1, 1, -1)
                _, (ht_new_t, ct_new_t) = self.target_lstm(target_disp_embedding,
                    (target_ht.view(1, 1, -1), target_ct.view(1, 1, -1)))
                target_ht = ht_new_t.view(1, -1)
                target_ct = ct_new_t.view(1, -1)

            # Nearby obstacles forward
            curr_N = torch.sum(nearby_ts_obs_mask).long().item()
            if curr_N >= 1:
                nearby_disp_embedding = self.nearby_rel_disp_embedding(
                    (curr_nearby_obs_pos_rel[:, nearby_ts_obs_mask,:].view(1, curr_N, -1)).clone()) \
                    .view(curr_N, 1, -1)

                _, (ht_new_n, ct_new_n) = self.nearby_lstm(nearby_disp_embedding,
                    (nearby_ht_list[nearby_ts_obs_mask, :].view(1, curr_N, -1),
                     nearby_ct_list[nearby_ts_obs_mask, :].view(1, curr_N, -1)))

                nearby_ht_list[nearby_ts_obs_mask, :] = ht_new_n.view(curr_N, 1, -1)
                nearby_ct_list[nearby_ts_obs_mask, :] = ct_new_n.view(curr_N, 1, -1)

                # Attention
                curr_nearby_pos = curr_nearby_obs_pos[:, nearby_ts_obs_mask, :]
                curr_target_pos = curr_target_obs_pos.repeat(1, curr_N, 1)
                curr_tn_rel_pos = curr_target_pos - curr_nearby_pos

                att_nearby_embedding = self.nearby_attention_embedding(
                    curr_tn_rel_pos.clone()).view(1, curr_N, -1)

                att_target_embedding = self.target_attention_embedding(
                    curr_target_obs_pos_rel.clone()).view(1, 1, -1)
                att_scores = torch.sum((att_nearby_embedding * att_target_embedding), 2)
                att_scores = att_scores * torch.tensor(curr_N) / \
                             torch.sqrt(torch.tensor(self.attention_dim).float())
                att_scores_numerator = torch.exp(att_scores)
                att_scores_denominator = torch.sum(att_scores_numerator, 1)
                att_scores = att_scores_numerator / att_scores_denominator
                att_scores = att_scores.view(att_scores.size(1), -1)

                Ht = (torch.sum((ht_new_n.view(curr_N, -1) * att_scores), 0)).view(1, -1)

        print(pred_out)
        return pred_out
