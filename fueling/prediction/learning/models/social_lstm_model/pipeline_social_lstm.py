#!/usr/bin/env python

import argparse

import torch
from torch.utils.data import DataLoader

from fueling.learning.train_utils import train_valid_dataloader
from fueling.prediction.learning.datasets.public_human_trajectory_dataset.human_trajectory_dataset \
    import collate_scenes
from fueling.prediction.learning.models.social_lstm_model.social_lstm_model import (
    HumanTrajectoryDataset,
    ProbablisticTrajectoryLoss,
    SocialLSTM,
)


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(description='social-lstm model training pipeline')

    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')

    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = HumanTrajectoryDataset(args.train_file, obs_len=8, pred_len=12,
                                           skip=1, min_ped=0, delim='\t')
    valid_dataset = HumanTrajectoryDataset(args.valid_file, obs_len=8, pred_len=12,
                                           skip=1, min_ped=0, delim='\t')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=8, collate_fn=collate_scenes)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True,
                              num_workers=8, collate_fn=collate_scenes)

    # Model and training setup
    model = SocialLSTM()
    loss = ProbablisticTrajectoryLoss()
    print(model)
    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=2, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    # Model training:
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=100, save_name='./', print_period=1)
