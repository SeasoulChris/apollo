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


def LabelCleaning(feature_dir, label_dir, pred_len=3.0):
    # From feature_dir, locate those labels of interests.

    # Get the statistics of all the labels. (histogram of how many left-turns,
    # right-turns, u-turns, go-straight, etc.)

    # Go through all labels of interests, filter out those noisy ones and
    # only retain those clean ones.
        # 1. Only keep pred_len length
        # 2. Get the scalar acceleration, and angular speed of all points.
        # 3. Fit a (3rd order?) polynomial to acc and ang-speed.
        # 4. Those with large residual errors should be removed.

    # Get the statistics of the cleaned labels, and do some re-balancing to
    # maintain roughly the same distribution as before.

    return
