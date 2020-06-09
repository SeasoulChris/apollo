#!/usr/bin/env python

import copy

import torch
import torch.nn as nn

from learning_algorithms.prediction.models.social_lstm_model.social_lstm_model import *


grid_size = 2
embed_size = 64
hidden_size = 128


class SingleLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(SingleLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers=1,
                            batch_first=True)

    def forward(self, X):
        et_at = X[:, : 2 * self.embed_size]
        ht = X[:, -2 * self.hidden_size: -self.hidden_size]
        ct = X[:, -self.hidden_size:]
        _, (out_ht, out_ct) = self.lstm(et_at.view(1, 1, -1),
                                        (ht.view(1, 1, -1), ct.view(1, 1, -1)))
        return out_ht, out_ct


if __name__ == '__main__':

    social_lstm = SocialLSTM()
    social_lstm.load_state_dict(torch.load("social_lstm_model.pt", map_location='cpu'))

    pos_eb = social_lstm.pos_embedding
    pos_eb.eval()
    traced_pos_eb = torch.jit.trace(pos_eb, torch.zeros([1, 2]))
    traced_pos_eb.save("./traced_pos_eb.pt")

    social_eb = social_lstm.social_embedding
    social_eb.eval()
    social_eb_input_size = grid_size * grid_size * hidden_size
    traced_social_eb = torch.jit.trace(social_eb, torch.zeros([1, social_eb_input_size]))
    traced_social_eb.save("./traced_social_eb.pt")

    single_lstm = SingleLSTM(embed_size, hidden_size)
    single_lstm.lstm = copy.deepcopy(social_lstm.lstm)
    single_lstm.eval()
    traced_single_lstm = torch.jit.trace(single_lstm,
                                         torch.zeros([1, 2 * (embed_size + hidden_size)]))
    traced_single_lstm.save("./traced_single_lstm.pt")

    pred_layer = social_lstm.pred_layer
    pred_layer.eval()
    traced_pred_layer = torch.jit.trace(pred_layer, torch.zeros([1, hidden_size]))
    traced_pred_layer.save("./traced_pred_layer.pt")
