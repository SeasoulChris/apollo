#!/usr/bin/env python

import argparse
import os

import torch

from fueling.learning.train_utils import train_valid_dataloader
from fueling.prediction.learning.pipelines.pedestrian_trajectory_prediction \
    .pedestrian_trajectory_dataset import PedestrianTrajectoryDataset
from fueling.prediction.learning.models.semantic_map_model.semantic_map_model \
    import SemanticMapSelfLSTMModel, WeightedSemanticMapLoss


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save_path', type=str, default='/fuel/',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    train_dataset = PedestrianTrajectoryDataset(args.train_file)
    valid_dataset = PedestrianTrajectoryDataset(args.valid_file)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                               num_workers=16, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True,
                                               num_workers=16, drop_last=True)

    model = SemanticMapSelfLSTMModel(30, 20)
    loss = WeightedSemanticMapLoss()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer, scheduler,
                           epochs=50, save_name=args.save_path, print_period=200, save_mode=2)
