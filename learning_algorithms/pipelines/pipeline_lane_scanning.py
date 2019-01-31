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

from data_preprocessing.features_labels_combining import *
from models.lane_scanning_model import *
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

    args = parser.parse_args()

    # load and preprocess data:
    # TODO(jiacheng): currently for test-purpose only, must replace with new
    #                 code later.
    data_train_all = LoadDataForLearning(args.train_file)
    data_train = []
    for item in data_train_all:
        data_train.append(item.features_for_learning)
    data_train = preprocess_features(data_train)

    data_valid_all = LoadDataForLearning(args.valid_file)
    data_valid = []
    for item in data_valid_all:
        data_valid.append(item.features_for_learning)
    data_valid = preprocess_features(data_valid)

    X_train = data_train
    y_train = X_train[:,-20:]
    X_valid = data_valid
    y_valid = X_valid[:,-20:]

    # Model and training setup
    model = lane_scanning_model()
    loss = lane_scanning_loss()
    print (model)
    learning_rate = 1e-1
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
                        train_batch=1024, print_period=100, valid_batch=1024)
