#!/usr/bin/env python

import argparse

import torch

# TODO: Avoid import *
from fueling.learning.train_utils import *
from fueling.learning.loss_utils import *
from learning_algorithms.prediction.datasets.apollo_vehicle_trajectory_dataset \
    .apollo_vehicle_trajectory_dataset import *
from learning_algorithms.prediction.models.lane_attention_trajectory_model \
    .lane_attention_trajectory_model import *
from learning_algorithms.prediction.models.semantic_map_model.semantic_map_model import *


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    IMG_MODE = True

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save_path', type=str, default='/fuel/',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    # Set-up data-loader
    train_dataset = ApolloVehicleTrajectoryDataset(args.train_file, IMG_MODE)
    valid_dataset = ApolloVehicleTrajectoryDataset(args.valid_file, IMG_MODE)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                               num_workers=16, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True,
                                               num_workers=16, drop_last=True)

    # Model and training setup
    # model = SelfLSTM()
    # model = SemanticMapSelfLSTMModel(30, 20)
    model = SemanticMapSelfAttentionLSTMModel(30, 20)
    loss = SemanticMapLoss()

    # model = SemanticMapSelfLSTMMultiModal(30, 20, num_modes=2)
    # loss = TrajectoryBivariateGaussianLoss()

    def base_loss_fn(y_pred, y_true):
        diff = y_pred[:, :, :2] - y_true
        diff = torch.sqrt(torch.sum(diff ** 2, 2))
        out = torch.mean(diff, dim=1)
        return out

    def base_loss_info(y_pred, y_true):
        diff = y_pred[:, :, :2] - y_true
        diff = torch.sqrt(torch.sum(diff ** 2, 2))
        out = torch.mean(diff)
        return out

    # loss = MultiModalLoss(base_loss_fn, base_loss_info)

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
