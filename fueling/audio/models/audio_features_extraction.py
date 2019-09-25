#!/usr/bin/env python

import argparse
import os

from absl import flags
import numpy as np
import librosa
from tqdm import tqdm
from scipy.signal import butter, lfilter, hilbert
from sklearn.utils import shuffle

from fueling.audio.pyAudioAnalysis import audioFeatureExtraction
from fueling.common import file_utils
from fueling.common.learning.train_utils import *


def preprocess(y):
    y_filt = butter_bandpass_filter(y)
    analytic_signal = hilbert(y_filt)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def butter_bandpass_filter(data, lowcut=500, highcut=1500, fs=8000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


class AudioFeatureExtraction(object):
    def __init__(self, data_dir, win_size=16, step=8):
        self.data_dir = data_dir
        self.win_size = win_size
        self.step = step
        self.pos_features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.pos_labels = []  # 1: emergency, 0: non-emergency
        self.neg_features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.neg_labels = []  # 1: emergency, 0: non-emergency

        self.features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.labels = []  # 1: emergency, 0: non-emergency

    def extract_cnn_features(self):
        files = file_utils.list_files(self.data_dir)
        for file in tqdm(files):
            if file.find('.wav') == -1:
                continue
            try:
                signal, sr = librosa.load(file, sr=8000)
            except:
                print("Failed to open file {}".format(file))
                continue
            signal = preprocess(signal)
            S = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)  # [128, len]

            label = 1
            if file.find("nonEmergency") != -1:
                label = 0

            start = 0
            while start + self.win_size <= log_S.shape[1]:
                end = start + self.win_size
                log_S_segment = log_S[:, start:end]
                if label == 1:
                    self.pos_features.append(log_S_segment)
                    self.pos_labels.append(label)
                else:
                    self.neg_features.append(log_S_segment)
                    self.neg_labels.append(label)
                start += self.step
        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels

    def extract_mlp_features(self):
        files = file_utils.list_files(self.data_dir)
        for file in tqdm(files):
            if file.find('.wav') == -1:
                continue
            try:
                signal, sr = librosa.load(file, sr=8000)
            except:
                print("Failed to open file {}".format(file))
                continue
            signal = preprocess(signal)
            total_features = audioFeatureExtraction.stFeatureExtraction(
                signal, sr, 0.10*sr, .05*sr)

            label = 1
            if file.find("nonEmergency") != -1:
                label = 0

            if label == 1:
                self.pos_features.extend(total_features)
                self.pos_labels.extend([label] * len(total_features))
            else:
                self.neg_features.extend(total_features)
                self.neg_labels.extend([label] * len(total_features))

        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels

    def balance_features(self, enable_shuffle=True):
        min_features_size = min(len(self.pos_features), len(self.neg_features))
        self.pos_features = self.pos_features[:min_features_size]
        self.pos_labels = self.pos_labels[:min_features_size]
        self.neg_features = self.neg_features[:min_features_size]
        self.neg_labels = self.neg_labels[:min_features_size]

        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels

        if len(self.features) != len(self.labels):
            raise ValueError('features size not matching labels size')

        if enable_shuffle:
            shuffle(self.features, self.labels, random_state=0)

    def save_features(self, model_type, features_dir):
        features = np.array(self.features)
        labels = np.array(self.labels)
        np.save(os.path.join(features_dir, model_type + '_features.npy'), features)
        np.save(os.path.join(features_dir, model_type + '_labels.npy'), labels)

    def load_features_labels(self, model_type, features_dir):
        features = np.load(os.path.join(
            features_dir, model_type + '_features.npy'))
        labels = np.load(os.path.join(
            features_dir, model_type + '_labels.npy'))

        return features, labels


if __name__ == "__main__":

    flags.DEFINE_string(
        'feature_type', 'mlp',
        'Feature type for training from [mlp, cnn].')

    flags.DEFINE_string(
        'train_dir', '/home/xukecheng/Desktop/cleaned_data/train_balanced/',
        'The dirname with training data.')

    flags.DEFINE_string(
        'valid_dir', '/home/xukecheng/Desktop/cleaned_data/eval_balanced/',
        'The dirname with validation data.')

    def main(argv):

        # data parser:
        flags_dict = flags.FLAGS.flag_values_dict()
        feature_type = flags_dict['feature_type']
        train_dir = flags_dict['train_dir']
        valid_dir = flags_dict['valid_dir']

        # train set features extraction and save
        train_set_extractor = AudioFeatureExtraction(train_dir)

        if feature_type == 'cnn':
            train_set_extractor.extract_cnn_features()
        elif feature_type == 'mlp':
            train_set_extractor.extract_mlp_features()
        else:
            raise ValueError(
                'model_type not properly defined, only support cnn or mlp')

        train_set_extractor.balance_features(True)

        train_set_extractor.save_features(feature_type, train_dir)

        # validation set features extraction and save
        validation_set_extractor = AudioFeatureExtraction(valid_dir)

        if feature_type == 'cnn':
            validation_set_extractor.extract_cnn_features()
        elif feature_type == 'mlp':
            validation_set_extractor.extract_mlp_features()
        else:
            raise ValueError(
                'model_type not properly defined, only support cnn or mlp')

        validation_set_extractor.save_features(feature_type, valid_dir)

    from absl import app
    app.run(main)
