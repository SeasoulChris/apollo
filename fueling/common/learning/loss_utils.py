#!/usr/bin/env python
"""Loss utils."""

import math

import numpy as np
import torch
import torch.nn as nn


class TrajectoryBivariateGaussianLoss:
    def loss_fn(self, y_pred, y_true, eps=1e-5):
        if y_pred is None:
            return cuda(torch.tensor(0))
        # y_pred: N x pred_len x 5
        # y_true: N x pred_len x 2
        mux, muy, sigma_x, sigma_y, corr = y_pred[:, :, 0].float(), y_pred[:, :, 1].float(),\
            y_pred[:, :, 2].float(), y_pred[:, :, 3].float(), y_pred[:, :, 4].float()
        x, y = y_true[:, :, 0].float(), y_true[:, :, 1].float()
        N = y_pred.size(0)
        if N == 0:
            return cuda(torch.tensor(0))

        corr = torch.clamp(corr, min=-1 + eps, max=1 - eps)
        z = (x - mux)**2 / (sigma_x**2 + eps) + (y - muy)**2 / (sigma_y**2 + eps) - 2 * \
            corr * (x - mux) * (y - muy) / (torch.sqrt((sigma_x * sigma_y)**2) + eps)
        z = torch.clamp(z, min=eps)

        P = 1 / (2 * np.pi * torch.sqrt((sigma_x * sigma_y)**2) *
                 torch.sqrt(1 - corr**2) + eps) * torch.exp(-z / (2 * (1 - corr**2)))

        loss = torch.clamp(P, min=eps)
        loss = -loss.log()

        return torch.sum(loss) / N

    def loss_info(self, y_pred, y_true):
        diff = y_pred[:, :, :2] - y_true
        diff = torch.sqrt(torch.sum(diff ** 2, 2))
        out = torch.mean(diff)
        return out


'''
y_pred: N x num_of_modes x sequence_length x 2 (or 5)
y_true: N x sequence_length x 2 (ground_truth_x, ground_truth_y)
'''
class MultiModalLoss:
    def __init__(self, base_loss_fn, base_loss_info):
        self.base_loss_fn_ = base_loss_fn
        self.base_loss_info_ = base_loss_info

    def loss_fn(self, y_pred, y_true):
        N = y_pred.size(0)
        num_modes = y_pred.size(1)
        loss_matrix = torch.zeros(N, num_modes).cuda()
        for i in range(num_modes):
            y_pred_curr_mode = y_pred[:, i, :, :]
            loss_curr_mode = self.base_loss_fn_(y_pred_curr_mode, y_true)
            loss_matrix[:, i] = loss_curr_mode

        final_loss = torch.min(loss_matrix, dim=1)
        final_loss = torch.mean(final_loss)

        return final_loss

    def loss_info(self, y_pred, y_true):
        N = y_pred.size(0)
        num_modes = y_pred.size(1)
        loss_matrix = torch.zeros(N, num_modes).cuda()
        for i in range(num_modes):
            y_pred_curr_mode = y_pred[:, i, :, :]
            loss_curr_mode = self.base_loss_info_(y_pred_curr_mode, y_true)
            loss_matrix[:, i] = loss_curr_mode

        final_loss_info = torch.min(loss_matrix, dim=1)
        final_loss_info = torch.mean(final_loss_info)

        return final_loss_info
