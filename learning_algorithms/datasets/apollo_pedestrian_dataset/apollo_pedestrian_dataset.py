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

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

import data_for_learning_pb2
from data_for_learning_pb2 import *


def LoadDataForLearning(filepath):
    list_of_data_for_learning = data_for_learning_pb2.ListDataForLearning()
    with open(filepath, 'rb') as file_in:
        list_of_data_for_learning.ParseFromString(file_in.read())
    return list_of_data_for_learning.data_for_learning


class ApolloPedestrianDataset(Dataset):
    def __init__(self, data_dir, obs_len=20, pred_len=40, threshold_dist_to_adc=20.0,
                 verbose=False):
        all_file_paths = os.listdir(data_dir)
        all_file_paths = [os.path.join(data_dir, _path) for _path in all_file_paths]
        seq_len = obs_len + pred_len

        # Go through all files, save pedestrian data by their IDs.


    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx
