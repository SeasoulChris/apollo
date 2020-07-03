#!/usr/bin/env python
"""Network utils."""

import cv2 as cv
import torch
import torch.nn as nn


def generate_cnn1d(dim_list):
    return


def generate_mlp(dim_list, last_layer_nonlinear=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    if not last_layer_nonlinear:
        layers = layers[:-2]
    return nn.Sequential(*layers)


def generate_lstm_states(hidden_size, bilateral=False):
    h0 = torch.zeros(1, hidden_size)
    c0 = torch.zeros(1, hidden_size)
    nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
    h0 = nn.Parameter(h0, requires_grad=True)
    c0 = nn.Parameter(c0, requires_grad=True)
    return h0, c0


def generate_lstm(input_size, hidden_size, bilateral=False):
    h0 = torch.zeros(1, hidden_size)
    c0 = torch.zeros(1, hidden_size)
    nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
    h0 = nn.Parameter(h0, requires_grad=True)
    c0 = nn.Parameter(c0, requires_grad=True)
    lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    return h0, c0, lstm


@torch.jit.script
def clip_boxes(xval, yval, feature_shape):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """
    Clip boxes to image boundaries.
    """
    lower_bound = torch.zeros_like(xval)
    upper_bound_x = torch.ones_like(xval) * (feature_shape[1] - 1)
    upper_bound_y = torch.ones_like(yval) * (feature_shape[0] - 1)
    # x1 >= 0
    xval = torch.max(torch.min(xval, upper_bound_x), lower_bound)
    # y1 >= 0
    yval = torch.max(torch.min(yval, upper_bound_y), lower_bound)
    return (xval, yval)


def rotate(img, angle):
    center = (img.shape[0] / 2, img.shape[1] / 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (img.shape[0], img.shape[1]))
    return (rotated, M)


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
