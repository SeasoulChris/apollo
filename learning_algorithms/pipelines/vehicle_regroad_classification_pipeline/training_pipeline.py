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

import argparse
import logging
import numpy as np
import os
import torch

from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *
from learning_algorithms.datasets.apollo_vehicle_regroad_dataset.apollo_vehicle_regroad_dataset import *
from learning_algorithms.models.lane_attention_model.lane_attention_model import *


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = ApolloVehicleRegularRoadDataset(args.train_file, \
    	training_mode=True, cutin_augmentation_coeff=2)
    valid_dataset = ApolloVehicleRegularRoadDataset(args.valid_file, \
    	training_mode=False, cutin_augmentation_coeff=2)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,\
        num_workers=8, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True,\
        num_workers=8, drop_last=True, collate_fn=collate_fn)

    # Model and training setup
    model = FastLaneAttention()
    loss = ClassificationLoss()
    print (model)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-11, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
    else:
        print ("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=100, save_name='./', print_period=10)
