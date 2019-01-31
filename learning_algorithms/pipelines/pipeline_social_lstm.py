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

from models.social_lstm.social_lstm_model import *
from utilities.IO_utils import *
from utilities.train_utils import *

if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description='social-lstm model training pipeline')

    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')

    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = HumanTrajectoryDataset(args.train_file, obs_len=8, pred_len=12,\
        skip=1, min_ped=0, delim='\t')
    valid_dataset = HumanTrajectoryDataset(args.valid_file, obs_len=8, pred_len=12,\
        skip=1, min_ped=0, delim='\t')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,\
        num_workers=8, collate_fn=collate_scenes)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True,\
        num_workers=8, collate_fn=collate_scenes)

    # Model and training setup
    model = SocialLSTM()
    loss = ProbablisticTrajectoryLoss()
    print (model)
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=2, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
    else:
        print ("Not using CUDA.")


    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=100, save_name='./', print_period=1)
