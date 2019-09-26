#!/usr/bin/env python

import argparse
import os

from absl import flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from fueling.common import file_utils
from fueling.common.learning.train_utils import *
from fueling.audio.models.audio_features_extraction import AudioFeatureExtraction
from learning_algorithms.prediction.datasets.apollo_vehicle_trajectory_dataset.apollo_vehicle_trajectory_dataset import *
from learning_algorithms.prediction.models.lane_attention_trajectory_model.lane_attention_trajectory_model import *
from learning_algorithms.prediction.models.semantic_map_model.semantic_map_model import *


class AudioDataset(Dataset):
    def __init__(self, mode, features, labels):
        self.mode = mode
        # a list of spectrograms, each: [n_mels, win_size]
        self.features = features
        self.labels = labels  # 1: emergency, 0: non-emergency

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'mlp':
            label = np.float32(self.labels[idx])
            feature = torch.from_numpy(self.features[idx]).float()
            return (feature, label)
        if self.mode == 'cnn1d':
            label = np.float32(self.labels[idx])
            return (torch.from_numpy(self.features[idx]), label)
        if self.mode == 'cnn2d':
            img = torch.from_numpy(self.features[idx])
            h = img.size(0)
            w = img.size(1)
            img = img.view(1, h, w).clone()
            label = np.float32(self.labels[idx])
            return (img, label)


class AudioLoss():
    def loss_fn(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        loss_func = nn.BCELoss()
        return loss_func(y_pred, y_true)

    def loss_info(self, y_pred, y_true):
        tag_pred = (y_pred > 0.5)
        tag_true = (y_true > 0.5)
        tag_pred = tag_pred.view(-1)
        tag_true = tag_true.view(-1)
        accuracy = (tag_pred == tag_true).type(torch.float).mean()
        return accuracy


class AudioMLPModel(nn.Module):
    def __init__(self, input_dim=105):
        super(AudioMLPModel, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 105)
        self.fc2 = nn.Linear(105, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)
        self.batchnorm = nn.BatchNorm1d(self.input_dim)

    def forward(self, X):
        X = self.batchnorm(X)
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        out = torch.sigmoid(self.fc3(X))
        return out


class AudioCNN1dModel(nn.Module):
    def __init__(self):
        super(AudioCNN1dModel, self).__init__()
        self.conv1 = nn.Conv1d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 4, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, X):
        # Conv layers
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = F.relu(self.conv3(X))
        # Flatten
        X = X.view(-1, 16 * 4)
        # FC layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = torch.sigmoid(self.fc3(X))

        return X


class AudioCNN2dModel(nn.Module):
    def __init__(self):
        super(AudioCNN2dModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*2*32, 100)
        self.fc2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, X):
        # Conv layers
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = self.pool(F.relu(self.conv3(X)))
        # Flatten
        X = X.view(-1, 16*2*32)
        # FC layers
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = torch.sigmoid(self.fc2(X))

        return X


if __name__ == "__main__":

    flags.DEFINE_string(
        'model_type', 'mlp',
        'Model type for training from [mlp, cnn1d, cnn2d].')

    flags.DEFINE_string(
        'train_dir', '/home/jinyun/cleaned_data/train_balanced/',
        'The dirname with training data.')

    flags.DEFINE_string(
        'valid_dir', '/home/jinyun/cleaned_data/eval_balanced/',
        'The dirname with validation data.')

    flags.DEFINE_string(
        'model_dir', './',
        'The dirname to save trained models.')

    def main(argv):

        # Set-up the GPU to use
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # data parser:
        flags_dict = flags.FLAGS.flag_values_dict()
        model_type = flags_dict['model_type']
        feature_type = 'mlp'
        if model_type == 'cnn1d' or model_type == 'cnn2d':
            feature_type = 'cnn'

        train_dir = flags_dict['train_dir']
        valid_dir = flags_dict['valid_dir']
        model_dir = flags_dict['model_dir']

        # Set-up data-loader
        train_features, train_labels = AudioFeatureExtraction.load_features_labels(
            feature_type, train_dir)
        train_dataset = AudioDataset(model_type, train_features, train_labels)

        valid_features, valid_labels = AudioFeatureExtraction.load_features_labels(
            feature_type, valid_dir)
        valid_dataset = AudioDataset(model_type, valid_features, valid_labels)

        print('--------- Loading Training Data -----------')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                                   num_workers=8, drop_last=True)
        print('--------- Loading Validation Data -----------')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True,
                                                   num_workers=8, drop_last=True)

        # Model and training setup
        model = None
        if model_type == 'cnn1d':
            model = AudioCNN1dModel()
        elif model_type == 'cnn2d':
            model = AudioCNN2dModel()
        elif model_type == 'mlp':
            model = AudioMLPModel()
        print('------ Model Structure -------')
        print(model)

        loss = AudioLoss()

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
        train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                               scheduler, epochs=50, save_name='./', print_period=100)

    from absl import app
    app.run(main)
