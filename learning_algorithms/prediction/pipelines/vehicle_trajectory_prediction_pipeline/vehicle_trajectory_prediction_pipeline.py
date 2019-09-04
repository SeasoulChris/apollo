#!/usr/bin/env python

import argparse
import torch

from learning_algorithms.utilities.train_utils import *
from learning_algorithms.prediction.datasets.apollo_vehicle_trajectory_dataset.apollo_vehicle_trajectory_dataset import *
from learning_algorithms.prediction.models.lane_attention_trajectory_model.lane_attention_trajectory_model import *
from learning_algorithms.prediction.models.semantic_map_model.semantic_map_model import *


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = ApolloVehicleTrajectoryDataset(args.train_file)
    valid_dataset = ApolloVehicleTrajectoryDataset(args.valid_file)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_fn)

    # Model and training setup
    model = SelfLSTM()
    loss = ProbablisticTrajectoryLoss()

    # print(model)
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
    else:
        print ("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=50, save_name='./', print_period=50)
