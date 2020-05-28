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
import fueling.common.logging as logging

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

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU()
        )

        self.memory_fc_layer = nn.Sequential(
            nn.Linear(100 * 100, 256),
            nn.ReLU()
        )

        self.input_fc_layer = nn.Sequential(
            nn.Linear(256 + 1000, 256),
            nn.ReLU()
        )

        self.input_deconv_layer = nn.Sequential(
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
            nn.Linear(self.input_img_size_h * self.input_img_size_w, 4),
            nn.ReLU()
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        # M_B_k = torch.cat((X[1], X[2]), dim=1)
        # M_B_k = nn.Parameter(M_B_k, requires_grad=True)
        # original code above, changed to below for jit trace success
        # TODO(Jinyun): check possible issue without nn.Parameter
        M_B_k = torch.cat((X[1], X[2]), dim=1).to(img_feature.device)

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
            P_B_k = self.memory_fc_layer(P_B_k)

            F_P_B_k = torch.cat((img_feature_encoding, P_B_k), dim=1)
            F_P_B_k = self.input_fc_layer(F_P_B_k)
            F_P_B_k = F_P_B_k.view(batch_size, 1, 16, 16)
            F_P_B_k = self.input_deconv_layer(F_P_B_k)

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
            # arg_max_row_index = arg_max_index // F_P_B_k.shape[-2:][0]
            # arg_max_col_index = arg_max_index % F_P_B_k.shape[-2:][1]
            # original code above, changed to below for jit trace success
            # TODO(Jinyun): TracerWarning: torch.tensor results are registered 
            # as constants in the trace. You can safely ignore this warning if 
            # you use this function to create tensors out of constant variables 
            # that would be the same every time you call this function. In any 
            # other case, this might cause the trace to be incorrect
            arg_max_row_index = arg_max_index // torch.tensor(
                [F_P_B_k.shape[-2:][0]]).to(img_feature.device)
            arg_max_col_index = arg_max_index % torch.tensor(
                [F_P_B_k.shape[-2:][1]]).to(img_feature.device)
            M_k_next = M_B_k[:, 0, :, :].clone()
            B_k_next = F_P_B_k[:, 1, :, :].clone()
            # TODO(Jinyun): TracerWarning: Converting a tensor to a Python index 
            # might cause the trace to be incorrect. We can't record the data flow
            # of Python values, so this value will be treated as a constant in the 
            # future. This means that the trace might not generalize to other inputs! 
            # like "for i in range(batch_size)" and "M_k_next[i, arg_max_row_index[i]"
            for i in range(batch_size):
                M_k_next[i, arg_max_row_index[i], arg_max_col_index[i]] = 1

            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        return (pred_pos_dists, pred_boxs, pred_points)


class TrajectoryImitationRNNMoreConvModel(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True, pred_horizon=10):
        super(TrajectoryImitationRNNMoreConvModel, self).__init__()
        # TODO(Jinyun): compressed to 3 channel, to be refined
        self.feature_compression_layer = nn.Conv2d(12, 3, 3, padding=1)

        self.feature_net = cnn_net(pretrained=pretrained)
        for param in self.feature_net.parameters():
            param.requires_grad = True

        self.feature_net_out_size = 1000
        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.memory_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.memory_fc_layer = nn.Sequential(
            nn.Linear(25 * 25, 256),
            nn.ReLU()
        )

        self.input_fc_layer = nn.Sequential(
            nn.Linear(256 + 1000, 256),
            nn.ReLU()
        )

        self.input_deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=2,
                               kernel_size=55, padding=10, stride=1),
            # size 16 to 50,
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=2,
                               kernel_size=71, padding=10, stride=1),
            # size 50 to 100,
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2, out_channels=2,
                               kernel_size=121, padding=10, stride=1),
            # size 100 to self.input_img_size_h,
            nn.ReLU()
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(25 * 25, 4),
            nn.ReLU()
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)
        M_B_k = nn.Parameter(M_B_k, requires_grad=True)

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
            P_B_k = self.memory_fc_layer(P_B_k)

            F_P_B_k = torch.cat((img_feature_encoding, P_B_k), dim=1)
            F_P_B_k = self.input_fc_layer(F_P_B_k)
            F_P_B_k = F_P_B_k.view(batch_size, 1, 16, 16)
            F_P_B_k = self.input_deconv_layer(F_P_B_k)

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
                M_k_next[i, arg_max_row_index[i], arg_max_col_index[i]] = 1

            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        return (pred_pos_dists, pred_boxs, pred_points)


class UnetDecoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(UnetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class TrajectoryImitationRNNUnetResnet18Model(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True, pred_horizon=10):
        super(TrajectoryImitationRNNUnetResnet18Model, self).__init__()

        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.base_model = models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=(7, 7), stride=(
                2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.decode2 = UnetDecoder(128, 128 + 64, 128)
        self.decode1 = UnetDecoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False)
        )

        self.output_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=121,
                      padding=10),  # size self.input_img_size_h to 100
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=71,
                      padding=10),  # size 100 to 50
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=46,
                      padding=10),  # size 50 to 25
            nn.ReLU()
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(25 * 25, 4),
            nn.ReLU()
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)
        M_B_k = nn.Parameter(M_B_k, requires_grad=True)

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
            stacked_imgs = torch.cat((img_feature, M_B_k), dim=1)

            e1 = self.layer1(stacked_imgs)  # 64,100,100
            e2 = self.layer2(e1)  # 64,50,50
            e3 = self.layer3(e2)  # 128,25,25
            d2 = self.decode2(e3, e2)  # 128,50,50
            d1 = self.decode1(d2, e1)  # 64,100,100
            F_P_B_k = self.decode0(d1)  # 64,200,200

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
                M_k_next[i, arg_max_row_index[i], arg_max_col_index[i]] = 1

            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        return (pred_pos_dists, pred_boxs, pred_points, M_B_k)


class TrajectoryImitationRNNTest(nn.Module):
    def __init__(self, input_img_size,
                 cnn_net=models.mobilenet_v2, pretrained=True, pred_horizon=10):
        super(TrajectoryImitationRNNTest, self).__init__()

        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.base_model = models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=(7, 7), stride=(
                2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.layer6 = self.base_layers[8]
        self.decode2 = UnetDecoder(128, 128 + 64, 128)
        self.decode1 = UnetDecoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False)
        )

        self.output_fc_layers = nn.Sequential(
            nn.Linear(512, 4),
        )

    def forward(self, X):
        img_feature = X[0]
        batch_size = img_feature.size(0)
        M_B_k = torch.cat((X[1], X[2]), dim=1)
        M_B_k = nn.Parameter(M_B_k, requires_grad=True)

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
            stacked_imgs = torch.cat((img_feature, M_B_k), dim=1)

            e1 = self.layer1(stacked_imgs)  # 64,100,100
            e2 = self.layer2(e1)  # 64,50,50
            e3 = self.layer3(e2)  # 128,25,25

            output_e1 = e3.clone()
            output_e2 = self.layer4(output_e1)
            output_e3 = self.layer5(output_e2)
            output_e4 = self.layer6(output_e3)
            output_e4 = output_e4.view(batch_size, -1)
            pred_point = self.output_fc_layers(output_e4)
            pred_points[:, t, :] = pred_point.clone()

            d2 = self.decode2(e3, e2)  # 128,50,50
            d1 = self.decode1(d2, e1)  # 64,100,100
            P_B_k = self.decode0(d1)  # 2,200,200

            P_k = P_B_k[:, 0, :, :].clone()
            B_k = P_B_k[:, 1, :, :].clone()
            P_k = torch.softmax(
                P_k.view(batch_size, -1), dim=1).view(batch_size,
                                                      self.input_img_size_h,
                                                      self.input_img_size_w)
            B_k = torch.sigmoid(B_k)

            pred_pos_dists[:, t, 0, :, :] = P_k.clone()
            pred_boxs[:, t, 0, :, :] = B_k.clone()

            arg_max_index = torch.argmax(P_k.view(batch_size, -1), dim=1)
            arg_max_row_index = arg_max_index // P_k.shape[-2:][0]
            arg_max_col_index = arg_max_index % P_k.shape[-2:][1]
            M_k_next = M_B_k[:, 0, :, :].clone()
            B_k_next = B_k.clone()
            for i in range(batch_size):
                M_k_next[i, arg_max_row_index[i], arg_max_col_index[i]] = 1

            M_B_k = torch.stack((M_k_next, B_k_next), dim=1)

        return (pred_pos_dists, pred_boxs, pred_points, M_B_k)


class TrajectoryImitationRNNLoss():

    def __init__(self, pos_dist_loss_weight=1, box_loss_weight=1, pos_reg_loss_weight=1):
        self.pos_dist_loss_weight = pos_dist_loss_weight
        self.box_loss_weight = box_loss_weight
        self.pos_reg_loss_weight = pos_reg_loss_weight

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

        logging.info("pos_dist_loss is {}, box_loss is {}, pos_reg_loss is {}".
                     format(
                         pos_dist_loss,
                         box_loss,
                         pos_reg_loss))
        logging.info("weighted_pos_dist_loss is {}, weighted_box_loss is {},"
                     "weighted_pos_reg_loss is {}".
                     format(
                         self.pos_dist_loss_weight * pos_dist_loss,
                         self.box_loss_weight * box_loss,
                         self.pos_reg_loss_weight * pos_reg_loss))

        return self.pos_dist_loss_weight * pos_dist_loss + \
            self.box_loss_weight * box_loss + \
            self.pos_reg_loss_weight * pos_reg_loss

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


if __name__ == "__main__":
    # code snippet for model prining
    model = models.resnet18(True)

    total_param = 0
    for name, param in model.named_parameters():
        print(name)
        print(param.data.shape)
        print(param.data.view(-1).shape[0])
        total_param += param.data.view(-1).shape[0]
    print(total_param)

    for index, (name, child) in enumerate(model.named_children()):
        print(index)
        print(name)
        print(child)
