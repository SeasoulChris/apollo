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
    def __init__(self, u_dim, kernel_dim, batch_size, seq=100):
        """Network initialization"""
        super().__init__()
        self.encoder = Encoder(u_dim, kernel_dim)
        self.attn = ScaledDotProductAttention()
        self.k_mat = torch.rand(batch_size, seq, u_dim)
        self.v_mat = torch.rand(batch_size, seq, u_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        data = self.attn(data, self.k_mat, self.v_mat)
        data = torch.transpose(data, -2, -3)
        data = self.encoder(data)
        return data


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """Network initialization"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """ q: query; k: key; v: value"""
        # (Batch, seq, feature)
        # Check if key and query size are the same
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


class AttentionHead(nn.Module):

    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries)  # (Batch, Seq, Feature)
        K = self.key_tfm(keys)  # (Batch, Seq, Feature)
        V = self.value_tfm(values)  # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        output = self.attn(Q, K, V)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # TODO(Shu): refactor this part
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        # shrink to original feature size (?)
        self.projection = nn.Linear(d_feature * n_heads, d_model)

    def forward(self, queries, keys, values, mask=None):
        output = [attn(queries, keys, values, mask=mask)  # (Batch, Seq, Feature)
                  for i, attn in enumerate(self.attn_heads)]

        # reconcatenate
        output = torch.cat(output, dim=2)  # (Batch, Seq, D_Feature * n_heads)

        # Final linear operation
        output = self.projection(output)  # (Batch, Seq, D_Model)
        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout=0.1):
        super().__init__()

        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)

        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):

        # STEP 1
        att = self.attn_head(x, x, x, mask=mask)

        # STEP 2
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))

        # STEP 3
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)

        # STEP 4
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))
        return x


class TransformerEncoder(nn.Module):
    """ transformer encoder """

    def __init__(self, n_blocks, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_model // n_heads, n_heads=n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor, mask=None):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)

        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)

        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Step 1
        # Apply attention to inputs
        att = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(att))

        # Step 2
        # Apply attention to the encoder outputs and outputs of the previous layer
        att = self.attn_head(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(att))

        # Step 3
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout=0.1):
        super().__init__()
        # self.position_embedding = PositionalEmbedding(d_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_model // n_heads, n_heads=n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor,
                enc_out: torch.FloatTensor,
                src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


class TransformerEncoderCNN(nn.Module):
    def __init__(self, u_dim, kernel_dim, dim_head=1):
        """Network initialization"""
        super().__init__()
        self.d_head = dim_head
        self.d_model = u_dim * dim_head
        self.encoder = TransformerEncoder(n_blocks=1, d_model=self.d_model,
                                          d_ff=1024, n_heads=dim_head)
        self.conv1 = nn.Conv1d(self.d_model, 100, self.d_model, stride=4)
        self.conv2 = nn.Conv1d(100, 50, self.d_model, stride=4)
        self.fc = nn.Linear(250, kernel_dim)

    def forward(self, data):
        """Define forward computation and activation functions"""
        # original data shape: [sequency/window_size, batch_size, channel]
        encoded_data = self.encoder(data.repeat(1, 1, self.d_head))
        # conv_input shape: [batch_size, channel, sequency/window_size]
        conv1_input = torch.transpose(torch.transpose(encoded_data, -2, -3), -2, -1)
        data = F.relu(self.conv1(conv1_input))
        data = F.relu(self.conv2(data))
        data = self.fc(data.view(data.shape[0], -1))
        return data
