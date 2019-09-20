#!/usr/bin/env python

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from fueling.audio.Paper3.pyAudioAnalysis import audioBasicIO
from fueling.audio.Paper3.pyAudioAnalysis import audioFeatureExtraction
from fueling.common import file_utils


class AudioDataset(Dataset):
    def __init__(self, data_dir, mode='cnn1d', win_size=16, step=8):
    	self.mode = mode
    	self.win_size = win_size
    	self.step = step
        self.features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.labels = []  # 1: emergency, 0: non-emergency
        files = file_utils.list_files(data_dir)
        for file in files:
            signal, sr = librosa.load(fn, sr=8000)
            signal = preprocess(signal)
            S = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            log_S = log_S.t()  # shape: [128, len]
            label = 1
            if file.find("nonEmergency") != -1:
                label = 0
            start = 0
            while start + win_size <= log_S.shape[1]:
                end = start + win_size
                log_S_segment = log_S[:, start:end]
                features.append(log_S_segment)
                labels.append(label)
                start += step

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # TODO(all): maybe the type of label need to be modified
        if self.mode == 'cnn1d':
            return ((torch.from_numpy(self.features[idx])),
                    torch.from_numpy(self.labels[idx]*np.ones(1, 1)))
        if self.mode == 'cnn2d':
            img = torch.from_numpy(self.features[idx])
            h = img.size(0)
            w = img.size(1)
            img = img.view(h, w, 1).clone()
            label = torch.from_numpy(self.labels[idx]*np.ones(1, 1))
            return ((img), label)


class AudioLoss():
    def loss_fn(self, y_pred, y_true):
        # TODO(kechxu): implement
        pass

    def loss_info(self, y_pred, y_true):
        # TODO(kechxu): implement
        return



class AudioCNN1dModel(nn.Module):
    def __init__(self):
        super(AudioCNN1dModel, self).__init__()
        # TODO(jinyun): implement
        pass

    def forward(self, X):
        # TODO(jinyun): implement
        pass


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
        X = F.sigmoid(self.fc2(x))

        return X


if __name__ == "__main__":
    print("hello")
