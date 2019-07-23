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

from learning_algorithms.prediction.datasets.apollo_pedestrian_dataset.apollo_pedestrian_dataset import *
from learning_algorithms.prediction.models.social_interaction_model.human_trajectory_dataset import *
from learning_algorithms.prediction.models.social_interaction_model.social_interaction_model import *
from learning_algorithms.utilities.train_utils import *


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = HumanTrajectoryDataset(args.train_file, obs_len=6, pred_len=10,\
        skip=1, min_ped=0, delim='\t', extra_sample=3, noise_std_dev=0.05, verbose=True)
    # valid_dataset = HumanTrajectoryDataset(args.valid_file, obs_len=6, pred_len=10,\
    #     skip=1, min_ped=0, delim='\t', extra_sample=3, noise_std_dev=0.0, verbose=True)
    valid_dataset = ApolloPedestrianDataset(args.valid_file, threshold_dist_to_adc=30.0, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,\
        num_workers=8, drop_last=True, collate_fn=collate_scenes)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True,\
        num_workers=8, drop_last=True, collate_fn=collate_scenes)

    # Model and training setup
    model = SimpleLSTM(pred_len=40)
    loss = ProbablisticTrajectoryLoss()
    print (model)
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
    else:
        print ("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=200, save_name='./temp_trained_models', print_period=1)
