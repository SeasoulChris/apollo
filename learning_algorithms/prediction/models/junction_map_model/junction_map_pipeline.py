#!/usr/bin/env python

import argparse

import torch

from fueling.common.learning.train_utils import *
from learning_algorithms.prediction.models.junction_map_model.junction_map_model import *


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description='semantic_map model training pipeline')

    parser.add_argument('--train_dir', type=str, help='training data directory')
    parser.add_argument('--valid_dir', type=str, help='validation data directory')

    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = JunctionMapDataset(args.train_dir)
    valid_dataset = JunctionMapDataset(args.valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=16)

    # Model and training setup
    model = JunctionMapModel(20, 12)
    loss = JunctionMapLoss()
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
                           scheduler, epochs=20, save_name='./', print_period=10)

    torch.save(model.state_dict(), save_name + '/model.pt')
