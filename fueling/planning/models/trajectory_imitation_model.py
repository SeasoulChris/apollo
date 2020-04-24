#!/usr/bin/env python

import glob
import os

import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import models

from fueling.common.coord_utils import CoordUtils

'''
========================================================================
Model definition
========================================================================
'''


class TrajectoryImitationCNNModel(nn.Module):
    def __init__(self,
                 cnn_net=models.mobilenet_v2, pretrained=True, pred_horizon=10):
        super(TrajectoryImitationCNNModel, self).__init__()
        # compressed to 3 channel
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon

        self.fc = nn.Sequential(
            nn.Linear(self.feature_net_out_size, 500),
            nn.Dropout(0.3),
            nn.Linear(500, 120),
            nn.Dropout(0.3),
            nn.Linear(120, self.pred_horizon * 5)
        )

    def forward(self, X):
        img = X
        out = self.compression_cnn_layer(img)
        out = self.cnn(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.pred_horizon * 5)
        return out


class TrajectoryImitationCNNLoss():
    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        y_true = y_true.view(y_true.size(0), -1)
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        y_true = y_true.view(y_true.size(0), -1)
        out = y_pred - y_true
        out = torch.sqrt(torch.sum(out ** 2, 1))
        out = torch.mean(out)
        return out


class TrajectoryImitationRNNModel(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True, pred_horizon=10):
        super(TrajectoryImitationRNNModel, self).__init__()
        # TODO(Jinyun): compressed to 3 channel, to be refined
        self.feature_compression_layer = nn.Conv2d(12, 3, 3, padding=1)

        self.feature_net = cnn_net(pretrained=pretrained)
        for param in self.feature_net.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        # TODO(Jinyun): implementation and evaluate Method 1 below
        # Method 1:
        # a. use ConvTranspose2d to upsample feature([1000,])
        #    to [1, self.input_img_size_h, self.input_img_size_w]
        # b. stack with M_k-1([1, self.input_img_size_h, self.input_img_size_w]) and
        #    B_k-1([1, self.input_img_size_h, self.input_img_size_w])
        #    inorder to get 3 channel image as input
        # c. pass a Conv layer to get hidden state which is P_k and B_k
        # d. add P_k and B_k to output
        # e. pass P_k and B_k through output layer to get [x_k, y_k, phi, v]
        #    and add it to output
        # f. use argmax to update M_k by P_k
        # g. feed M_k and B_k to next iteration

        # Method 2:
        # a. use Conv2d and FC layer to encode M_k-1 and B_k-1
        #    ([2, self.input_img_size_h, self.input_img_size_w]) to ([256,])
        # b. stack with feature([1000,]) inorder to get a 1d vector as input
        # c. pass a ConvTranspose2d layer to output P_k and B_k
        # d. add P_k and B_k to output
        # e. pass P_k and B_k through output layer to get [x_k, y_k, phi, v]
        #    and add it to output
        # f. use argmax to update M_k by P_k
        # g. feed M_k and B_k to next iteration

        self.M_B_0 = torch.zeros(
            2, self.input_img_size_h, self.input_img_size_w)
        nn.init.xavier_normal_(
            self.M_B_0[1, :, :], gain=nn.init.calculate_gain('relu'))
        self.M_B_0 = nn.Parameter(self.M_B_0, requires_grad=True)

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU()
        )

        self.memory_flatener = nn.Sequential(
            nn.Linear(100 * 100, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
        )

        self.input_fc_layer = nn.Sequential(
            nn.Linear(256 + 1000, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU()
        )

        self.input_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=2,
                               kernel_size=70, padding=10, stride=10),
            # size 16 * 16 to self.input_img_size_h,
            nn.ReLU()
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(self.input_img_size_h * self.input_img_size_w, 1000),
            nn.ReLU(),
            nn.Linear(1000, 120),
            nn.ReLU(),
            nn.Linear(120, 4),
            nn.ReLU()
        )

    def forward(self, X):
        img_feature = X
        batch_size = img_feature.size(0)
        M_B_k = self.M_B_0.repeat(batch_size, 1, 1, 1)

        img_feature_encoding = self.feature_net(
            self.feature_compression_layer(img_feature))

        pred_pos_dists = torch.zeros(
            (batch_size, self.pred_horizon, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_boxs = torch.zeros(
            (batch_size, self.pred_horizon, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        pred_points = torch.zeros(
            (batch_size, self.pred_horizon, 4), device=img_feature.device)

        for t in range(self.pred_horizon):
            P_B_k = self.memory_encoder(M_B_k)
            P_B_k = P_B_k.view(batch_size, -1)
            P_B_k = self.memory_flatener(P_B_k)

            F_P_B_k = torch.cat((img_feature_encoding, P_B_k), dim=1)
            F_P_B_k = self.input_fc_layer(F_P_B_k)
            F_P_B_k = F_P_B_k.view(batch_size, 1, 16, 16)
            F_P_B_k = self.input_conv_layer(F_P_B_k)

            P_k = F_P_B_k[:, 0, :, :].clone()
            B_k = F_P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists[:, t, 0, :, :] = P_k
            pred_boxs[:, t, 0, :, :] = B_k

            pred_point = self.output_conv_layer(F_P_B_k)
            pred_point = pred_point.view(batch_size, -1)
            pred_point = self.output_fc_layers(pred_point)
            pred_points[:, t, :] = pred_point.clone()

            arg_max_index = torch.argmax(
                F_P_B_k[:, 0, :, :].view(batch_size, -1), dim=1)
            arg_max_row_index = arg_max_index // F_P_B_k.shape[-2:][0]
            arg_max_col_index = arg_max_index % F_P_B_k.shape[-2:][1]
            M_k_next = M_B_k[:, 0, :, :].clone()
            B_k_next = F_P_B_k[:, 1, :, :].clone()
            for i in range(batch_size):
                M_k_next[i, arg_max_row_index, arg_max_col_index] = 1

            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        return (pred_pos_dists, pred_boxs, pred_points)


class TrajectoryImitationRNNLoss():
    def loss_fn(self, y_pred, y_true):
        batch_size = y_pred[0].shape[0]
        pred_pos_dists = y_pred[0].view(batch_size, -1)
        pred_boxs = y_pred[1].view(batch_size, -1)
        pred_points = y_pred[2].view(batch_size, -1)
        true_pos_dists = y_true[0].view(batch_size, -1)
        true_boxs = y_true[1].view(batch_size, -1)
        true_points = y_true[2].view(batch_size, -1)

        pos_dist_loss = nn.BCELoss()(pred_pos_dists, true_pos_dists)

        box_loss = nn.BCELoss()(pred_boxs, true_boxs)

        pos_reg_loss = nn.L1Loss()(pred_points, true_points)

        return pos_dist_loss + box_loss + pos_reg_loss

    def loss_info(self, y_pred, y_true):
        # Focus on pose displacement error
        pred_points = y_pred[2]
        true_points = y_true[2]
        points_diff = pred_points - true_points
        pose_diff = points_diff[:, :, 0:2]

        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        return out


class TrajectoryImitationWithEnvRNNLoss():
    def loss_fn(self, y_pred, y_true):
        batch_size = y_pred[0].shape[0]
        pred_pos_dists = y_pred[0].view(batch_size, -1)
        pred_boxs = y_pred[1].view(batch_size, -1)
        pred_points = y_pred[2].view(batch_size, -1)
        true_pos_dists = y_true[0].view(batch_size, -1)
        true_boxs = y_true[1].view(batch_size, -1)
        true_points = y_true[2].view(batch_size, -1)
        true_pred_obs = y_true[3].view(batch_size, -1)
        true_offroad_mask = y_true[4].view(batch_size, -1)

        pos_dist_loss = nn.BCELoss()(pred_pos_dists, true_pos_dists)

        box_loss = nn.BCELoss()(pred_boxs, true_boxs)

        pos_reg_loss = nn.L1Loss()(pred_points, true_points)

        collision_loss = torch.mean(pred_boxs * true_pred_obs)

        offroad_loss = torch.mean(pred_boxs * true_offroad_mask)

        return pos_dist_loss + box_loss + pos_reg_loss + \
            collision_loss + offroad_loss

    def loss_info(self, y_pred, y_true):
        # Focus on pose displacement error
        pred_points = y_pred[2]
        true_points = y_true[2]
        points_diff = pred_points - true_points
        pose_diff = points_diff[:, :, 0:2]

        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        return out
