#!/usr/bin/env python

import argparse
import os

import torch

import fueling.common.logging as logging

from fueling.learning.train_utils import train_valid_dataloader
from fueling.planning.datasets.semantic_map_dataset import SemanticMapDataset
from fueling.planning.models.semantic_map_model import SemanticMapModel, SemanticMapLoss


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    IMG_MODE = True

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    logging.info('training directory:{} validation directory:{}'.format(args.train_file, args.valid_file))

    # Set-up data-loader
    train_dataset = SemanticMapDataset(args.train_file)
    valid_dataset = SemanticMapDataset(args.valid_file)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True,
                                               num_workers=4, drop_last=True)

    # Model and training setup
    model = SemanticMapModel(80, 20)
    loss = SemanticMapLoss()

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
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=10, save_name='./', print_period=50)
