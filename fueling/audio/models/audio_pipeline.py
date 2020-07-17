#!/usr/bin/env python

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from fueling.learning.train_utils import *
from fueling.audio.models.siren_net import SirenNet, SirenNetDataset, SirenNetLoss


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

    train_dataset = SirenNetDataset(args.train_file)
    valid_dataset = SirenNetDataset(args.valid_file)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True,
                                               num_workers=4, drop_last=True)

    model = SirenNet()
    loss = SirenNetLoss()
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
                           epochs=50, save_name=args.save_path, print_period=50, save_mode=2)
