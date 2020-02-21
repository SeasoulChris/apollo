#!/usr/bin/env python

import argparse

import torch

from fueling.learning.train_utils import *
from learning_algorithms.prediction.models.interaction_model.interaction_model import *


delta = 0.5
epochs = 5

dim_input = 3
dim_output = 1


def data_preprocessing(data):
    X = data[:, dim_input:] - data[:, :dim_input]

    n = X.shape[0]
    Y = np.zeros([n, 1])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


if __name__ == '__main__':

    # data parser:
    parser = argparse.ArgumentParser(
        description='semantic_map model training pipeline')

    parser.add_argument('--data', type=str, help='training data filename')

    parser.add_argument('-s', '--savepath', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()

    data = np.load(args.data)
    print("Data load success, with data shape: " + str(data.shape))

    train_data, test_data = train_test_split(data, test_size=0.2)
    X_train, Y_train = data_preprocessing(train_data)
    X_test, Y_test = data_preprocessing(test_data)

    print(X_train.shape)

    model = InteractionModel(dim_input, delta)
    loss = InteractionLoss()
    learning_rate = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=2, min_lr=1e-8, verbose=True, mode='min')

    # CUDA setup:
    if (torch.cuda.is_available()):
        print ("Using CUDA to speed up training.")
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    else:
        print ("Not using CUDA.")

    # Model training:
    model = train_valid_vanilla(X_train, Y_train, X_test, Y_test, model, loss,
                                optimizer, scheduler, epochs, 'interaction_model.pt',
                                train_batch=65536)
    for p in model.parameters():
        print(p)
    savepath = args.savepath + "interaction_model.pt"
    traced_script_module = torch.jit.trace(model, X_train[0:1])
    traced_script_module.save(savepath)
