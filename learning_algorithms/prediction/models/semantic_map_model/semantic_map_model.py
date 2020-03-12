#!/usr/bin/env python

import glob
import os

import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from fueling.common.coord_utils import CoordUtils
from fueling.learning.network_utils import *
from fueling.learning.train_utils import *
from learning_algorithms.prediction.models.semantic_map_model.self_attention import Self_Attn
from learning_algorithms.prediction.models.semantic_map_model.spatial_attention import SpatialAttention2d

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
            label_dict = np.load(os.path.join(os.path.dirname(img_name)
                                              .replace("image-feature", "features-san-mateo-new")
                                              .replace("image-valid", "features-san-mateo-new"),
                                              'future_status.npy')).item()
            future_pos = label_dict[key]
            origin = future_pos[0]
            past_pos = [CoordUtils.world_to_relative(pos, origin) for pos in past_pos]
            future_pos = [CoordUtils.world_to_relative(pos, origin) for pos in future_pos]

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
        img = X[0]
        obs_pos = X[3]
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
        out = torch.sqrt(torch.sum(out ** 2, 2))
        out = torch.mean(out)
        return out


class SemanticMapSelfLSTMModel(nn.Module):
    def __init__(self, pred_len, observation_len,
                 embed_size=64, hidden_size=128,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(SemanticMapSelfLSTMModel, self).__init__()
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        self.pred_len = pred_len
        self.observation_len = observation_len
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
        img = X[0]
        obs_pos = X[3]
        obs_pos_step = X[4]
        N = obs_pos.size(0)
        ht, ct = self.h0.repeat(1, N, 1), self.h0.repeat(1, N, 1)

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        pred_traj = torch.zeros((N, self.pred_len, 2), device = img.device)

        for t in range(1, self.observation_len + self.pred_len):
            if t < self.observation_len:
                curr_obs_pos_step = obs_pos_step[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                pred_input = torch.cat((ht.view(N, -1), img_embedding), 1)
                curr_obs_pos_step = self.pred_layer(pred_input).float().clone()
                curr_obs_pos = curr_obs_pos + curr_obs_pos_step
                pred_traj[:, t - self.observation_len, :] = curr_obs_pos.clone()

            disp_embedding = self.disp_embed(curr_obs_pos_step.clone()).view(N, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        return pred_traj


class SemanticMapSelfAttentionLSTMModel(nn.Module):
    def __init__(self, pred_len, observation_len,
                 embed_size=64, hidden_size=128,
                 cnn_net=models.resnet50, pretrained=True):
        super(SemanticMapSelfAttentionLSTMModel, self).__init__()
        # self.att = Self_Attn(3)
        self.attn = SpatialAttention2d(3)
        self.cnn = nn.Sequential(*list(cnn_net(pretrained=pretrained).children())[:-1])
        self.cnn_out_size = 2048
        self.pred_len = pred_len
        self.observation_len = observation_len
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
        img = X[0]
        obs_pos = X[3]
        obs_pos_step = X[4]
        N = obs_pos.size(0)
        ht, ct = self.h0.repeat(1, N, 1), self.h0.repeat(1, N, 1)

        img, img_attn = self.attn(img)
        img_embedding = self.cnn(img_att)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        pred_traj = torch.zeros((N, self.pred_len, 2), device = img.device)

        for t in range(1, self.observation_len + self.pred_len):
            if t < self.observation_len:
                curr_obs_pos_step = obs_pos_step[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                pred_input = torch.cat((ht.view(N, -1), img_embedding), 1)
                curr_obs_pos_step = self.pred_layer(pred_input).float().clone()
                curr_obs_pos = curr_obs_pos + curr_obs_pos_step
                pred_traj[:, t - self.observation_len, :] = curr_obs_pos.clone()

            disp_embedding = self.disp_embed(curr_obs_pos_step.clone()).view(N, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        return pred_traj


class SemanticMapSelfLSTMModelWithUncertainty(nn.Module):
    def __init__(self, pred_len, observation_len,
                 embed_size=64, hidden_size=128,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(SemanticMapSelfLSTMModelWithUncertainty, self).__init__()
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        self.pred_len = pred_len
        self.observation_len = observation_len
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
            nn.Linear(hidden_size + self.cnn_out_size, 5),
        )

    def forward(self, X):
        img = X[0]
        obs_pos = X[3]
        obs_pos_step = X[4]
        N = obs_pos.size(0)
        ht, ct = self.h0.repeat(1, N, 1), self.h0.repeat(1, N, 1)

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        curr_obs_pos_step = torch.zeros((N, 2), device = img.device)
        pred_traj = torch.zeros((N, self.pred_len, 5), device = img.device)

        for t in range(1, self.observation_len + self.pred_len):
            if t < self.observation_len:
                curr_obs_pos_step = obs_pos_step[:, t, :].float()
                curr_obs_pos = obs_pos[:, t, :].float()
            else:
                pred_input = torch.cat((ht.view(N, -1), img_embedding), 1)
                curr_obs_pos_step = self.pred_layer(pred_input).float().clone()
                pred_traj[:, t - self.observation_len, :] = curr_obs_pos_step
                curr_obs_pos = curr_obs_pos + curr_obs_pos_step[:, 0:2]
                pred_traj[:, t - self.observation_len, 0:2] = curr_obs_pos


            disp_embedding = self.disp_embed(curr_obs_pos_step[:, 0:2]).view(N, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        return pred_traj

class SemanticMapSelfLSTMMultiModal(nn.Module):
    def __init__(self, pred_len, observation_len,
                 embed_size=64, hidden_size=128, num_modes=2,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(SemanticMapSelfLSTMMultiModal, self).__init__()
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        self.pred_len = pred_len
        self.observation_len = observation_len
        self.num_modes = num_modes
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.disp_embed = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0 = generate_lstm_states(hidden_size)
        self.lstm = [nn.LSTM(embed_size, hidden_size,
                             num_layers=1, batch_first=True)] * self.num_modes

        self.pred_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 2),
        )

    def forward(self, X):
        img = X[0]
        obs_pos = X[3]
        obs_pos_step = X[4]
        N = obs_pos.size(0)

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)
        pred_traj = torch.zeros((N, self.num_modes, self.pred_len, 2), device = img.device)

        for i in range(self.num_modes):
            ht, ct = self.h0.repeat(1, N, 1), self.h0.repeat(1, N, 1)
            self.lstm[i].to(device = img.device)
            for t in range(1, self.observation_len + self.pred_len):
                if t < self.observation_len:
                    curr_obs_pos_step = obs_pos_step[:, t, :].float()
                    curr_obs_pos = obs_pos[:, t, :].float()
                else:
                    pred_input = torch.cat((ht.view(N, -1), img_embedding), 1)
                    curr_obs_pos_step = self.pred_layer(pred_input).float().clone()
                    curr_obs_pos = curr_obs_pos + curr_obs_pos_step
                    pred_traj[:, i, t - self.observation_len, :] = curr_obs_pos.clone()
                disp_embedding = self.disp_embed(curr_obs_pos_step.clone()).view(N, 1, -1)

                _, (ht, ct) = self.lstm[i](disp_embedding, (ht, ct))

        return pred_traj

class SemanticMapSocialAttentionModel(nn.Module):
    '''
    Semantic map model with social attention
    '''
    def __init__(self, pred_len, num_history_points,
                 embed_size=64, edge_hidden_size=256, node_hidden_size=128, attention_dim=64,
                 cnn_net=models.mobilenet_v2, pretrained=True):
        super(SemanticMapSocialAttentionModel, self).__init__()
        self.pred_len = pred_len

        # CNN
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = self.cnn.fc.in_features
        fc_in_features = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.edge_hidden_size = edge_hidden_size

        self.step_disp_embedding = nn.Linear(2, embed_size)

        # Target hidden state
        self.target_h0, self.target_c0 = generate_lstm_states(node_hidden_size)

        # Target RNN flow
        self.target_lstm = nn.LSTM(embed_size, node_hidden_size,
                                   num_layers=1, batch_first=True)
        self.target_attention_embedding = nn.Linear(2, attention_dim)

        # Nearby hidden state
        self.nearby_h0, self.nearby_c0 = generate_lstm_states(edge_hidden_size)

        # Nearby RNN flow
        self.embed_size = embed_size
        self.nearby_lstm = nn.LSTM(embed_size, edge_hidden_size, num_layers=1, batch_first=True)
        self.nearby_attention_embedding = nn.Linear(2, attention_dim)

        self.attention_dim = attention_dim

        # Prediction FC layer
        self.pred_layer = nn.Linear(node_hidden_size + self.cnn_out_size + edge_hidden_size, 2)

    def forward(self, X):
        img, target_obs_pos_abs, target_obs_hist_size, target_obs_pos_rel, target_obs_pos_step, \
             nearby_obs_pos_abs, nearby_obs_hist_sizes, nearby_obs_pos_rel, nearby_obs_pos_step, \
             num_nearby_obs = X

        N = img.size(0)
        observation_len = target_obs_pos_abs.size(1)
        nearby_padding_size = nearby_obs_pos_abs.size(1)
        M = N * nearby_padding_size
        pred_mask = cuda(torch.ones(N))
        pred_traj = cuda(torch.zeros(N, self.pred_len, 2))

        img_embedding = self.cnn(img)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)  # (N, cnn_out_size)

        # Initialize target RNN
        target_ht = self.target_h0.repeat(N, 1)
        target_ct = self.target_c0.repeat(N, 1)

        # Initialize nearby RNN
        # (M, edge_hidden_size)
        nearby_ht_list = self.nearby_h0.repeat(M, 1)
        nearby_ct_list = self.nearby_c0.repeat(M, 1)

        Ht = self.nearby_h0.repeat(N, 1)  # (N, edge_hidden_size)

        scores = torch.zeros(N, nearby_padding_size)

        for t in range(1, observation_len+self.pred_len):
            if t < observation_len:
                target_ts_obs_mask = (target_obs_hist_size > observation_len - t).view(-1) # (N,)
                curr_target_obs_pos_abs = target_obs_pos_abs[:, t, :].float()  # (N, 2)
                curr_target_obs_pos_rel = target_obs_pos_rel[:, t, :].float()  # (N, 2)
                curr_target_obs_pos_step = target_obs_pos_step[:, t, :].float() # (N, 2)
                # (N, nearby_padding_size, 1)
                nearby_ts_obs_mask = (nearby_obs_hist_sizes > observation_len-t)
                # (N, nearby_padding_size)
                nearby_ts_obs_mask = nearby_ts_obs_mask[:, :, 0]
                # (N, nearby_padding_size, 2)
                curr_nearby_obs_pos_abs = nearby_obs_pos_abs[:, :, t, :].float()
                # (N, nearby_padding_size, 2)
                curr_nearby_obs_pos_rel = nearby_obs_pos_rel[:, :, t, :].float()
                # (N, nearby_padding_size, 2)
                curr_nearby_obs_pos_step = nearby_obs_pos_step[:, :, t, :].float()
            else:
                target_ts_obs_mask = (target_obs_hist_size > -1).view(-1)
                nearby_ts_obs_mask = cuda(torch.ones(N, nearby_padding_size, 1))
                pred_input = torch.cat((target_ht, img_embedding, Ht), 1)
                pred_out = self.pred_layer(pred_input).float().clone()
                curr_target_obs_pos_abs = curr_target_obs_pos_abs + pred_out
                curr_target_obs_pos_rel = curr_target_obs_pos_rel + pred_out
                curr_target_obs_pos_step = pred_out
                pred_traj[:, t-observation_len, :] = curr_target_obs_pos_rel.clone()

            # Target obstacles forward
            num_target_ts_obs = torch.sum(target_ts_obs_mask).long().item()
            # TODO(kechxu) figure out the following if condition
            if num_target_ts_obs == 0:
                continue
            target_disp_embedding = self.step_disp_embedding(
                curr_target_obs_pos_step[target_ts_obs_mask==1, :]
                .view(num_target_ts_obs, 1, -1).clone())

            _, (ht_new_t, ct_new_t) = self.target_lstm(target_disp_embedding,
                (target_ht[target_ts_obs_mask==1, :].view(1, num_target_ts_obs, -1),
                 target_ct[target_ts_obs_mask==1, :].view(1, num_target_ts_obs, -1)))
            target_ht[target_ts_obs_mask==1, :] = ht_new_t.view(num_target_ts_obs, -1)
            target_ct[target_ts_obs_mask==1, :] = ct_new_t.view(num_target_ts_obs, -1)

            # Nearby obstacles forward
            curr_N = torch.sum(nearby_ts_obs_mask).long().item()
            if curr_N == 0:
                continue

            curr_nearby_ts_mask = nearby_ts_obs_mask.view(-1).long()

            nearby_disp_embedding = self.step_disp_embedding(
                (curr_nearby_obs_pos_step.view(-1, 2).clone())[curr_nearby_ts_mask==1, :] \
                .clone()).view(curr_N, 1, -1)

            _, (ht_new_n, ct_new_n) = self.nearby_lstm(nearby_disp_embedding,
                (nearby_ht_list[curr_nearby_ts_mask==1, :].view(1, curr_N, -1),
                 nearby_ct_list[curr_nearby_ts_mask==1, :].view(1, curr_N, -1)))

            nearby_ht_list[curr_nearby_ts_mask==1, :] = ht_new_n.view(curr_N, -1)
            nearby_ct_list[curr_nearby_ts_mask==1, :] = ct_new_n.view(curr_N, -1)

            # (M, 2)
            curr_nearby_pos_abs = curr_nearby_obs_pos_abs.view(-1, 2)
            # (M, 2)
            curr_target_pos_abs = curr_target_obs_pos_abs.repeat(1, nearby_padding_size).view(-1, 2)
            # (M, 2)
            curr_tn_rel_pos = curr_nearby_pos_abs - curr_target_pos_abs

            # (M, embed_size)
            att_nearby_embedding = self.nearby_attention_embedding(
                curr_tn_rel_pos.clone())

            att_target_embedding = self.target_attention_embedding(
                curr_target_obs_pos_rel.clone())
            # (M, embed_size)
            att_target_embedding = att_target_embedding \
                .repeat(1, nearby_padding_size).view(-1, self.embed_size)

            # (M,)
            att_scores = torch.sum((att_nearby_embedding * att_target_embedding), 1).clone()
            # (M,)
            att_scores = att_scores * torch.tensor(curr_N) / \
                         torch.sqrt(torch.tensor(self.attention_dim).float())
            att_scores = att_scores * (curr_nearby_ts_mask.float())
            att_score_max, _ = torch.max(att_scores.view(N, -1), 1)
            att_scores = att_scores.view(N, -1) - att_score_max.repeat(nearby_padding_size, 1).t()

            att_scores_numerator = (torch.exp(att_scores).view(-1) * \
                                    (curr_nearby_ts_mask.float())).view(N, -1)

            att_scores_denominator = \
                torch.sum(att_scores_numerator, 1).repeat(nearby_padding_size, 1).t() + 1e-6
            att_scores_final = (att_scores_numerator / \
                                att_scores_denominator).view(N, nearby_padding_size, 1) \
                         .repeat(1, 1, self.edge_hidden_size)

            Ht[:, :] = torch.sum(
                (nearby_ht_list.view(N, nearby_padding_size, -1).clone() * att_scores_final), 1)

        return pred_traj
