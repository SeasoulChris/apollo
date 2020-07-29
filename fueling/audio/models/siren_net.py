#!/usr/bin/env python

from collections import OrderedDict
from fnmatch import fnmatch
import os

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

from fueling.common import file_utils
import fueling.common.logging as logging


class Urbansound8K(object):
    def __init__(self, data_dir, sample_rate=22050, length=1.5, stride=0.5):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.length = length
        self.stride = 0.5
        self.signal_length = int(sample_rate * length)
        self.signal_stride = int(sample_rate * stride)

    def label_file(self, filepath, dest_pos_dir, dest_neg_dir):
        label = 0
        if fnmatch(filepath, '*-8-*-*.wav'):
            label = 1
        if filepath.find('ambulance') != -1 or filepath.find('firetruck') != -1:
            label = 1
        signals = librosa.load(filepath, sr=self.sample_rate)
        if signals is None:
            logging.info('None signals found')
            return None, None
        signals = signals[0]
        total_len = signals.shape[0]
        logging.info('total length = {}'.format(total_len))
        if total_len < self.signal_length:
            logging.info('Total length is too short')
            return None, None
        features = []
        for i in range(0, total_len - self.signal_length + 1, self.signal_stride):
            feature = signals[i:(i + self.signal_length)]
            feature = feature.reshape(1, self.signal_length)
            features.append((feature, label))
        return features, label

    def preprocess(self):
        files = file_utils.list_files(self.data_dir)
        pos_file_count = 0
        neg_file_count = 0
        pos_data_count = 0
        neg_data_count = 0
        for file in files:
            if file.find('.wav') == -1:
                continue
            logging.info('--- Dealing with {} ---'.format(file))
            origin_dir = os.path.dirname(file)
            dest_pos_dir = origin_dir.replace('audio', 'features/positive', 1)
            dest_neg_dir = origin_dir.replace('audio', 'features/negative', 1)
            os.makedirs(dest_pos_dir, exist_ok=True)
            os.makedirs(dest_neg_dir, exist_ok=True)
            features, label = self.label_file(file, dest_pos_dir, dest_neg_dir)
            if features is None or label is None:
                logging.info('Skip none data')
                continue
            dest_dir = None
            if label == 1:
                pos_file_count += 1
                pos_data_count += len(features)
                dest_dir = dest_pos_dir
            else:
                neg_file_count += 1
                neg_data_count += len(features)
                dest_dir = dest_neg_dir
            _, origin_file_name = os.path.split(file)
            dest_file_name = origin_file_name.replace('wav', 'npy', 1)
            np.save(os.path.join(dest_dir, dest_file_name), features)
            logging.info('File (pos, neg) = ({}, {})'.format(pos_file_count, neg_file_count))
            logging.info('Data (pos, neg) = ({}, {})'.format(pos_data_count, neg_data_count))


class SirenNetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        files = file_utils.list_files(data_dir)
        self.all_features = []
        for file in files:
            features = np.load(file, allow_pickle=True).tolist()
            for feature in features:
                self.all_features.append(feature)

    def __len__(self):
        pos, neg = 0, 0
        for feature in self.all_features:
            if (feature[1] > 0.5):
                pos += 1
            else:
                neg += 1
        print("Got a total of " + str(len(self.all_features)) + " data points, with " + str(pos)
              + " positive, and " + str(neg) + " negative labels.")
        return len(self.all_features)

    def __getitem__(self, idx):
        feature = self.all_features[idx]
        data_for_learn = feature[0]
        label = np.zeros((2))
        if feature[1] < 0.5:
            label[0] = 1
        else:
            label[1] = 1
        return ((torch.from_numpy(data_for_learn)), torch.from_numpy(label))


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
            ('output', nn.Linear(64, 2)),
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
            ('output', nn.Linear(64, 2)),
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
        eps = 1e-8
        wavenet_out = self.wavenet(X)
        mfcc_out = self.mfcc(X)
        log_mel_out = torch.log(self.mel_spec(X) + eps)
        spec_features = torch.cat([mfcc_out, log_mel_out], -2)
        mlnet_out = self.mlnet(spec_features)
        out = (wavenet_out + mlnet_out) / 2.0
        return out


class SirenNetLoss():
    def loss_fn(self, y_pred, y_true):
        true_label = y_true.topk(1)[1].view(-1)
        loss_func = nn.CrossEntropyLoss()
        return loss_func(y_pred, true_label)

    def loss_info(self, y_pred, y_true):
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        pred_label = y_pred.topk(1)[1]
        true_label = y_true.topk(1)[1]
        accuracy = (pred_label == true_label).type(torch.float).mean().item()
        print("Accuracy is {:.3f} %".format(100 * accuracy))
        return accuracy


if __name__ == '__main__':
    # input = torch.ones([1, 1, 33075])
    # model = SirenNet()
    # print(model(input))

    urbansound = Urbansound8K('/data/UrbanSound8K/audio/')
    urbansound.preprocess()
