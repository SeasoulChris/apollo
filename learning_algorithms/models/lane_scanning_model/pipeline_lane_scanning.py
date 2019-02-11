###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import logging
import numpy as np
import os
import torch
import sklearn

from data_preprocessing.features_labels_utils import *
from models.lane_scanning.lane_scanning_model import *
from utilities.IO_utils import *
from utilities.train_utils import *

cuda_is_available = torch.cuda.is_available()
logging.basicConfig(filename='testlog.log', level=logging.INFO)

if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description='lane scanning model training pipeline')

    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    parser.add_argument('-v', '--vanilla_train', action='store_true', \
                        help='Don\'t use data loader')

    args = parser.parse_args()

    # load and preprocess data:
    files_train = GetListOfFiles(args.train_file)
    files_valid = GetListOfFiles(args.valid_file)

    if args.vanilla_train:
        X_train, y_train = None, None
        for i, file in enumerate(files_train):
            file_content = np.load(file)
            X_temp, y_temp = preprocess_features(file_content)
            if X_train is None:
                X_train, y_train = X_temp, y_temp
            else:
                X_train = np.concatenate((X_train, X_temp), 0)
                y_train = np.concatenate((y_train, y_temp), 0)
            print ('Processed {} out of {} training files.'\
                   .format(i+1, len(files_train)))
        print (X_train.shape)
        print (y_train.shape)
        X_train, _, y_train, _ = sklearn.model_selection.train_test_split(\
            X_train, y_train, test_size=0.0, random_state=36)

        X_valid, y_valid = None, None
        for i, file in enumerate(files_valid):
            file_content = np.load(file)
            X_temp, y_temp = preprocess_features(file_content)
            if X_valid is None:
                X_valid, y_valid = X_temp, y_temp
            else:
                X_valid = np.concatenate((X_valid, X_temp), 0)
                y_valid = np.concatenate((y_valid, y_temp), 0)
            print ('Processed {} out of {} validation files.'\
                   .format(i+1, len(files_valid)))
        print (X_valid.shape)
        print (y_valid.shape)

        # Model and training setup
        model = lane_scanning_model()
        loss = lane_scanning_loss()
        print (model)
        learning_rate = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=2, min_lr=1e-9, verbose=True, mode='min')

        # CUDA setup:
        if (cuda_is_available):
            print ("Using CUDA to speed up training.")
            model.cuda()
            X_train = Variable(torch.FloatTensor(X_train).cuda())
            X_valid = Variable(torch.FloatTensor(X_valid).cuda())
            y_train = Variable(torch.FloatTensor(y_train).cuda())
            y_valid = Variable(torch.FloatTensor(y_valid).cuda())
        else:
            print ("Not using CUDA.")
            X_train = Variable(torch.FloatTensor(X_train))
            X_valid = Variable(torch.FloatTensor(X_valid))
            y_train = Variable(torch.FloatTensor(y_train))
            y_valid = Variable(torch.FloatTensor(y_valid))

        # Model training:
        train_valid_vanilla(X_train, y_train, X_valid, y_valid, model, loss,
                            optimizer, scheduler, 50, './', batch_preprocess,
                            train_batch=512, print_period=100, valid_batch=512)
    else:
        train_dataset = LaneScanningDataset(args.train_file)
        valid_dataset = LaneScanningDataset(args.valid_file)

        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True,\
            num_workers=8, collate_fn=collate_with_padding)
        valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True,\
            num_workers=8, collate_fn=collate_with_padding)

        # Model and training setup
        model = lane_scanning_model()
        loss = lane_scanning_loss()
        print (model)
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
                               scheduler, epochs=100, save_name='./', print_period=10)





