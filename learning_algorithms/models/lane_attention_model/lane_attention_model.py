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

import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset

from learning_algorithms.utilities.train_utils import *


class LaneAttention(nn.Module):
	def __init__(self):
		super(LaneAttention, self).__init__()
		self.vehicle_encoding = None
		self.lane_encoding = None
		self.lane_aggregation = None
		self.prediction_layer = None

	def forward(self, X):
		return X


class VehicleLSTM(nn.Module):
	def __init__(self):
		super(VehicleLSTM, self).__init__()

	def forward(self, X):
		return X


class LaneLSTM(nn.Module):
	def __init__(self):
		super(LaneLSTM, self).__init__()

	def forward(self, X):
		return X


class AttentionalAggregation(nn.Module):
	def __init__(self):
		super(AttentionalAggregation, self).__init__()

	def forward(self, X):
		return X
