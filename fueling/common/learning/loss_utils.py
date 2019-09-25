#!/usr/bin/env python
"""Loss utils."""

import math

import torch
import torch.nn as nn


'''
y_pred: N x sequence_length x 5 (mu_x, mu_y, sigma_x, sigma_y, correlation_xy)
y_true: N x sequence_length x 2 (ground_truth_x, ground_truth_y)
'''


def traj_bivariate_gaussian_loss(y_pred, y_true):
    # Sanity checks.
    if y_pred is None:
        return 0
    if y_pred.size(0) == 0:
        return 0

    N = y_pred.size(0)
    mux, muy, sigma_x, sigma_y, corr = y_pred[:, :, 0], y_pred[:, :, 1],\
        y_pred[:, :, 2], y_pred[:, :, 3], y_pred[:, :, 4]
    gt_x, gt_y = y_true[:, :, 0], y_true[:, :, 1]
    eps = 1e-20

    z = ((gt_x - mux) / (eps + sigma_x))**2 + ((gt_y - muy) / (eps + sigma_y))**2 - \
        2 * corr * (gt_x - mux) * (gt_y - muy) / (sigma_x * sigma_y + eps)
    P = 1 / (2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - corr**2) + eps) * \
        torch.exp(-z / (2 * (1 - corr**2)))

    loss = torch.clamp(P, min=eps)
    loss = -loss.log()

    return torch.sum(loss) / N


'''
y_pred: N x num_of_modes x y_true_dim[1:]
y_true: N x sequence_length x 2 (ground_truth_x, ground_truth_y)
'''


def multi_modal_loss(y_pred, y_true):
    N = y_pred.size(0)
    num_mode = y_pred.size(1)
    loss_matrix = torch.zeros(N, num_mode).cuda()
    loss_fn = nn.MSELoss(reduction='none')
    for i in range(y_pred.size(1)):
        y_pred_curr = y_pred[:, i, :]
        loss_curr = loss_fn(y_pred_curr, y_true)
        loss_curr = torch.mean(loss_curr, dim=1)
        loss_matrix[:, i] = loss_curr

    min_loss_idx = torch.argmin(loss_matrix, dim=1)
    final_loss = loss_matrix[:, min_loss_idx]
    final_loss = torch.mean(final_loss)

    return final_loss
