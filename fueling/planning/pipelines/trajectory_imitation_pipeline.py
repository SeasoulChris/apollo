#!/usr/bin/env python

import argparse
import os

import torch

import fueling.common.logging as logging

from fueling.learning.train_utils import train_valid_dataloader
from fueling.planning.datasets.img_in_traj_out_dataset import TrajectoryImitationCNNDataset, TrajectoryImitationRNNDataset
from fueling.planning.models.trajectory_imitation_model import TrajectoryImitationCNNModel, TrajectoryImitationCNNLoss, TrajectoryImitationRNNModel, TrajectoryImitationRNNLoss


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('model_type', type=str, help='model type, cnn or rnn')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    logging.info(
        'training directory:{} validation directory:{}'.format(
            args.train_file,
            args.valid_file))

    # Set-up data-loader
    model = None
    loss = None
    train_dataset = None
    valid_dataset = None

    if args.model_type == 'cnn':
        train_dataset = TrajectoryImitationCNNDataset(args.train_file)
        valid_dataset = TrajectoryImitationCNNDataset(args.valid_file)
        model = TrajectoryImitationCNNModel()
        loss = TrajectoryImitationCNNLoss()

    elif args.model_type == 'rnn':
        train_dataset = TrajectoryImitationRNNDataset(args.train_file)
        valid_dataset = TrajectoryImitationRNNDataset(args.valid_file)
        model = TrajectoryImitationRNNModel()
        loss = TrajectoryImitationRNNLoss()

    else:
        logging.info('model {} is not implemnted'.format(args.model_type))
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True,
                                               num_workers=4, drop_last=True)

    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    # Model training:
    torch.autograd.set_detect_anomaly(True)
    
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=10, save_name='./', print_period=50)
