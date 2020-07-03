#!/usr/bin/env python

import argparse

import torch
import argparse

import torch

from fueling.common.learning.train_utils import *
from fueling.common.learning.loss_utils import *
from semantic_map_single_image_dataset import *
from semantic_map_single_image_model import *
import cv2 as cv


# for documentation: http://wiki.baidu.com/display/AutoDrive/semantic+map+object+tracking
# single image trajectory prediction

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # resolve pytorch multithread data loader issue and cv.resize got stuck issue
    cv.setNumThreads(0)

    # data parser:
    parser = argparse.ArgumentParser(
        description='semantic_map single image model training pipeline')

    parser.add_argument('--train_dir', type=str, help='training data directory')
    parser.add_argument('--valid_dir', type=str, help='validation data directory')

    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = ApolloSinglePredictionTrajectoryDataset(args.train_dir)
    valid_dataset = ApolloSinglePredictionTrajectoryDataset(args.valid_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True,
                                               num_workers=10, drop_last=True,
                                               collate_fn=custom_collate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=10,
                                               shuffle=True, num_workers=10,
                                               drop_last=True, collate_fn=custom_collate)

    # Model and training setup
    model = TrajectoryPredictionSingle(10, 20)  # predict length followed by observation length
    loss = TrajectoryPredictionSingleLoss()
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=2, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=100, save_name='./', print_period=1)
