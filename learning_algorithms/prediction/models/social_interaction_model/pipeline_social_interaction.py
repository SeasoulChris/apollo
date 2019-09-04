#!/usr/bin/env python

import argparse

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
    train_dataset = HumanTrajectoryDataset(args.train_file, obs_len=6, pred_len=10,
                                           skip=1, min_ped=0, delim='\t', extra_sample=3, noise_std_dev=0.05, verbose=True)
    # valid_dataset = HumanTrajectoryDataset(args.valid_file, obs_len=6, pred_len=10,\
    #     skip=1, min_ped=0, delim='\t', extra_sample=3, noise_std_dev=0.0, verbose=True)
    valid_dataset = ApolloPedestrianDataset(
        args.valid_file, threshold_dist_to_adc=30.0, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_scenes)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True,
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
