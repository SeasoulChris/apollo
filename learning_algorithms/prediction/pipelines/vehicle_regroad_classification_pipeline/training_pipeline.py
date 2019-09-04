#!/usr/bin/env python

import argparse
import torch

from learning_algorithms.prediction.datasets.apollo_vehicle_regroad_dataset.apollo_vehicle_regroad_dataset import *
from learning_algorithms.prediction.models.lane_attention_model.lane_attention_model import *
from learning_algorithms.utilities.network_utils import *
from learning_algorithms.utilities.train_utils import *


def train_using_given_model_and_params(model_params, train_file, valid_file):
    # Set-up data-loader
    train_dataset = ApolloVehicleRegularRoadDataset(train_file,
                                                    training_mode=True, cutin_augmentation_coeff=2)
    valid_dataset = ApolloVehicleRegularRoadDataset(valid_file,
                                                    training_mode=False, cutin_augmentation_coeff=2)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True,
                              num_workers=8, drop_last=True, collate_fn=collate_fn)

    # Set-up model, optimizer, and scheduler
    model = FastLaneAttention(*model_params)
    loss = ClassificationLoss()
    print (model)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-11, verbose=True, mode='min')

    # Set-up CUDA
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
    else:
        print ("Not using CUDA.")

    # Model training
    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=100, save_name='./', print_period=10)


if __name__ == "__main__":
    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('mode', type=int, help='model mode')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    model_mode = [args.mode]
    train_using_given_model_and_params(model_mode, args.train_file, args.valid_file)
