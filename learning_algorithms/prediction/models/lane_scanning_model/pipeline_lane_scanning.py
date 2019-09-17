#!/usr/bin/env python

import argparse

import torch

from fueling.common.learning.train_utils import *
from learning_algorithms.prediction.models.lane_scanning_model.lane_scanning_model import *
import fueling.common.file_utils as file_utils


cuda_is_available = torch.cuda.is_available()

if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description='lane scanning model training pipeline')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save_path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    parser.add_argument('-v', '--vanilla_train', action='store_true',
                        help='Don\'t use data loader')
    parser.add_argument('-f', '--hyperparam_file', type=str,
                        default='./hyperparams.npy',
                        help='The path of hyper-parameters.')
    parser.add_argument('-i', '--hp_idx', type=int, default=-1,
                        help='Specify which set of hyper-parameters to use.')
    args = parser.parse_args()

    # load and preprocess data:
    logging.basicConfig(filename=args.save_path + 'testlog.log',
                        level=logging.INFO)

    files_train = file_utils.list_files(args.train_file)
    files_valid = file_utils.list_files(args.valid_file)

    train_dataset = LaneScanningDataset(args.train_file)
    valid_dataset = LaneScanningDataset(args.valid_file)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True,
                              num_workers=8, collate_fn=collate_with_padding)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True,
                              num_workers=8, collate_fn=collate_with_padding)

    # Model and training setup
    model = None
    if args.hp_idx == -1:
        # Use default hyperparams.
        model = lane_scanning_model()
    else:
        # Use the loaded hyperparams.
        all_hp = np.load(args.hyperparam_file)
        hp = all_hp.item().get(args.hp_idx)
        model = lane_scanning_model(
            dim_cnn=hp['dim_cnn'],
            hidden_size=hp['hidden_size'],
            dim_lane_fc=hp['dim_lane_fc'],
            dim_obs_fc=hp['dim_obs_fc'],
            dim_traj_fc=hp['dim_traj_fc'])

    loss = lane_scanning_loss()
    logging.info(model)
    print (model)
    learning_rate = 1e-3
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
                           scheduler, epochs=100, save_name=args.save_path,
                           print_period=None, early_stop=10)
