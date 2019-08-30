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

import torch
import torch.nn as nn

'''  '''


def generate_cnn1d(dim_list):
    return


'''  '''


def generate_mlp(dim_list, last_layer_nonlinear=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    if last_layer_nonlinear:
        return nn.Sequential(*layers)
    else:
        return nn.Sequential(*layers[:-2])


'''  '''


def generate_lstm_states(hidden_size, bilateral=False):
    h0 = torch.zeros(1, hidden_size)
    c0 = torch.zeros(1, hidden_size)
    nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
    h0 = nn.Parameter(h0, requires_grad=True)
    c0 = nn.Parameter(c0, requires_grad=True)
    return h0, c0


'''  '''


def generate_lstm(input_size, hidden_size, bilateral=False):
    h0 = torch.zeros(1, hidden_size)
    c0 = torch.zeros(1, hidden_size)
    nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
    h0 = nn.Parameter(h0, requires_grad=True)
    c0 = nn.Parameter(c0, requires_grad=True)
    lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    return h0, c0, lstm
