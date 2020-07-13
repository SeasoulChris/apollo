#!/usr/bin/env python

from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchaudio


class SirenNetDataset(Dataset):
    def __init__(self, data_dir):
        pass  # TODO(kechxu)

    def __len__(self):
        pass  # TODO(kechxu)

    def __getitem__(self, idx):
        pass  # TODO(kechxu)


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.raw_feature_extraction = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, 64, 64, padding=32)),
            ('bn1', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv1d(64, 64, 64, padding=31)),
            ('bn2', nn.BatchNorm1d(64)),
            ('relu', nn.ReLU()),
            ('pool1', nn.MaxPool1d(220))
        ]))
        self.classification = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(1, 32, kernel_size=4, padding=2)),
            ('bn3', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv4', nn.Conv2d(32, 32, kernel_size=4, padding=1)),
            ('bn4', nn.BatchNorm2d(32)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv5', nn.Conv2d(32, 64, kernel_size=4, padding=2)),
            ('bn5', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('conv6', nn.Conv2d(64, 64, kernel_size=4, padding=1)),
            ('bn6', nn.BatchNorm2d(64)),
            ('pool3', nn.MaxPool2d(2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(37888, 512)),
            ('dropout1', nn.Dropout(0.5)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(512, 64)),
            ('dropout2', nn.Dropout(0.5)),
            ('relu', nn.ReLU()),
            ('output', nn.Linear(64, 3)),
            ('softmax', nn.Softmax(dim=1)),
        ]))

    def forward(self, X):
        raw_features = self.raw_feature_extraction(X)
        raw_features.unsqueeze_(1)
        out = self.classification(raw_features)
        return out


class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=4, padding=2)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 32, kernel_size=4, padding=1)),
            ('bn2', nn.BatchNorm2d(32)),
            ('pool1', nn.MaxPool2d(2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(53248, 512)),
            ('dropout1', nn.Dropout(0.5)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(512, 64)),
            ('dropout2', nn.Dropout(0.5)),
            ('relu', nn.ReLU()),
            ('output', nn.Linear(64, 3)),
            ('softmax', nn.Softmax(dim=1)),
        ]))

    def forward(self, X):
        out = self.cnn(X)
        return out


class SirenNet(nn.Module):
    def __init__(self):
        super(SirenNet, self).__init__()
        self.wavenet = WaveNet()
        self.mlnet = MLNet()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=22050,
                                               melkwargs={"hop_length": 512})
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                             hop_length=512, n_mels=64)

    def forward(self, X):
        wavenet_out = self.wavenet(X)
        mfcc_out = self.mfcc(X)
        log_mel_out = torch.log(self.mel_spec(X))
        spec_features = torch.cat([mfcc_out, log_mel_out], -2)
        mlnet_out = self.mlnet(spec_features)
        out = (wavenet_out + mlnet_out) / 2.0
        return out


if __name__ == '__main__':
    input = torch.ones([1, 1, 33075])
    model = SirenNet()
    print(model(input))
