#!/usr/bin/env python

import argparse
import os

import numpy as np
import librosa
from tqdm import tqdm
from scipy.signal import butter, lfilter, hilbert

from fueling.audio.Paper3.pyAudioAnalysis import audioBasicIO
from fueling.audio.Paper3.pyAudioAnalysis import audioFeatureExtraction
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


class AudioFeatureExtraction():
    def __init__(self, data_dir, win_size=16, step=8):
        self.data_dir = data_dir
        self.win_size = win_size
        self.step = step
        self.features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.labels = []  # 1: emergency, 0: non-emergency

    def extract_features(self):
        files = file_utils.list_files(self.data_dir)
        for file in tqdm(files):
            if file.find('.wav') == -1:
                continue
            signal, sr = librosa.load(file, sr=8000)
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
                self.features.append(log_S_segment)
                self.labels.append(label)
                start += self.step

    def save_features(self, features_dir):
        features = np.array(self.features)
        labels = np.array(self.labels)
        np.save(os.path.join(features_dir, 'features.npy'), features)
        np.save(os.path.join(features_dir, 'labels.npy'), labels)

    def load_features_labels(self, features_dir):
        features = np.load(os.path.join(features_dir, 'features.npy'))
        labels = np.load(os.path.join(features_dir, 'labels.npy'))

        return features, labels


if __name__ == "__main__":
    # data parser:
    parser = argparse.ArgumentParser(description='features_extraction')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    args = parser.parse_args()

    # train set features extraction and save
    train_set_extractor = AudioFeatureExtraction(args.train_file)

    train_set_extractor.extract_features()

    train_set_extractor.save_features(args.train_file)

    # validation set features extraction and save
    validation_set_extractor = AudioFeatureExtraction(args.valid_file)

    validation_set_extractor.extract_features()

    validation_set_extractor.save_features(args.valid_file)
