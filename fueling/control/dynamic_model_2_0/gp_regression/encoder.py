#!/usr/bin/env python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fueling.common.logging as logging


class Encoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        super().__init__()
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, stride=4)
        self.fc = nn.Linear(250, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # original data shape: [sequency/window_size, batch_size, channel]
        # conv_input shape: [batch_size, channel, sequency/window_size]
        conv1_input = torch.transpose(torch.transpose(data, -2, -3), -2, -1)
        data = F.relu(self.conv1(conv1_input))
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        return data


class DummyEncoder(nn.Module):
    """encoder (for place holder only)"""

    def __init__(self):
        """Network initialization"""
        super().__init__()

    def forward(self, data):
        """Define forward computation and activation functions"""
        return data


class DilatedEncoder(nn.Module):
    def __init__(self, u_dim, kernel_dim):
        """Network initialization"""
        super().__init__()
        # (5, 200)
        self.conv1 = nn.Conv1d(u_dim, 100, u_dim, dilation=5, stride=4)
        self.conv2 = nn.Conv1d(100, 50, u_dim, dilation=1, stride=4)
        # set 200 to 250 for dilation = 3 or 2
        self.fc = nn.Linear(200, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # original data shape: [sequency/window_size, batch_size, channel]
        # conv_input shape: [batch_size, channel, sequency/window_size]
        conv1_input = torch.transpose(torch.transpose(data, -2, -3), -2, -1)
        data = F.relu(self.conv1(conv1_input))
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        return data


class AttenEcoder(nn.Module):
    def __init__(self, u_dim, kernel_dim, batch_size=1, seq=100):
        """Network initialization"""
        super().__init__()
        self.encoder = Encoder(u_dim, kernel_dim)
        self.attn = ScaledDotProductAttention(batch_size, seq, u_dim)
        self.k_mat = torch.rand(batch_size, seq, u_dim)
        self.v_mat = torch.rand(batch_size, seq, u_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        data = self.attn(data, self.k_mat, self.v_mat)
        data = torch.transpose(data, -2, -3)
        logging.info(data.shape)
        data = self.encoder(data)
        return data


class ScaledDotProductAttention(nn.Module):
    def __init__(self, batch=1, seq=100, feature=6, dropout=0.1):
        """Network initialization"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch = batch
        self.seq = seq
        self.feature = feature

    def forward(self, q, k, v, mask=None):
        """ q: query; k: key; v: value"""
        # (Batch, seq, feature)
        # Check if key and query size are the same
        logging.info(q.shape)
        d_k = k.size(-1)
        if q.size(-1) != d_k:
            logging.error(f'query size {q.size(-1)} is different from key size {d_k}')

        # STEP 1: Calculate score
        # compute the dot product between queries and keys for each batch
        # and position in the sequence
        k = k.transpose(1, 2)
        score_attn = torch.bmm(q, k)  # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence for each batch

        # STEP 2: Divide by sqrt(Dk)
        # Scale the dot products by the dimensionality.
        # Normalize the weights across the sequence dimension
        score_attn = score_attn / math.sqrt(d_k)
        # (Note that since we transposed, the sequence and feature dimensions are switched)

        # STEP 3: Mask Optional
        # fill attention weights with 0s where padded
        if mask is not None:
            score_attn = score_attn.masked_fill(mask, 0)

        # STEP 4: Softmax
        score_attn = torch.exp(score_attn)
        score_attn = score_attn / score_attn.sum(dim=-1, keepdim=True)

        score_attn = self.dropout(score_attn)

        # STEP 5: Matmul with value matrix
        output = torch.bmm(score_attn, v)  # (Batch, Seq, Feature)
        return output
