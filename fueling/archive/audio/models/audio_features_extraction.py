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
from fueling.learning.train_utils import *


class AudioFeatureExtraction(object):
    def clear(self):
        self.pos_features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.pos_labels = []  # 1: emergency, 0: non-emergency
        self.neg_features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.neg_labels = []  # 1: emergency, 0: non-emergency
        self.features = []  # a list of spectrograms, each: [n_mels, win_size]
        self.labels = []  # 1: emergency, 0: non-emergency

    def __init__(self, data_dir, sr=8000):
        self.data_dir = data_dir
        self.sample_rate = sr
        self.clear()

    def butter_bandpass_filter(self, data, lowcut=500, highcut=1500, order=5):
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def preprocess(self, y):
        y_filt = self.butter_bandpass_filter(y)
        analytic_signal = hilbert(y_filt)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def extract_signal_segments(self, time_segment=1.0, time_step=0.1):
        self.clear()
        files = file_utils.list_files(self.data_dir)
        for file in tqdm(files):
            if file.find('.wav') == -1:
                continue
            try:
                signal, sr = librosa.load(file, sr=self.sample_rate)
            except:
                print("Failed to open file {}".format(file))
                continue

            label = 1
            if file.find("nonEmergency") != -1:
                label = 0

            # signal = self.preprocess(signal)
            signal_len = signal.shape[0]
            segments = []
            signal_step = int(self.sample_rate * time_step)
            signal_pos = 0
            signal_seg_len = int(self.sample_rate * time_segment)
            while signal_pos + signal_seg_len <= signal_len:
                segment = signal[signal_pos : (signal_pos + signal_seg_len)]
                if label == 1:
                    self.pos_features.append(segment)
                    self.pos_labels.append(label)
                else:
                    self.neg_features.append(segment)
                    self.neg_labels.append(label)
                signal_pos += signal_step

        self.features = self.pos_features + self.neg_features
        self.labels = self.pos_labels + self.neg_labels

    def extract_cnn_features(self):
        self.clear()
        signal_segments, labels = self.load_features_labels('signal', self.data_dir)
        for i in tqdm(range(signal_segments.shape[0])):
            signal = signal_segments[i]
            label = labels[i]
            # S = librosa.feature.melspectrogram(signal, sr=self.sample_rate, n_mels=128)
            # log_S = librosa.power_to_db(S, ref=np.max)
            if label == 1:
                # self.neg_features.append(log_S)
                self.pos_features.append(signal)
                self.pos_labels.append(label)
            elif label == 0:
                # self.neg_features.append(log_S)
                self.neg_features.append(signal)
                self.neg_labels.append(label)
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
            signal = self.preprocess(signal)
            total_features = audioFeatureExtraction.stFeatureExtraction(
                signal, sr, 0.10*sr, .05*sr)

            label = 1
            if file.find("nonEmergency") != -1:
                label = 0

            if label == 1:
                self.pos_features.extend(total_features)
                self.pos_labels.extend([label] * len(total_features))
            elif label == 0:
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

    @staticmethod
    def load_features_labels(model_type, features_dir):
        features = np.load(os.path.join(
            features_dir, model_type + '_features.npy'))
        labels = np.load(os.path.join(
            features_dir, model_type + '_labels.npy'))

        return features, labels


if __name__ == "__main__":

    flags.DEFINE_string(
        'feature_type', 'cnn',
        'Feature type for training from [signal, mlp, cnn].')

    flags.DEFINE_string(
        'train_dir', '/home/jinyun/cleaned_data/train_balanced/',
        'The dirname with training data.')

    flags.DEFINE_string(
        'valid_dir', '/home/jinyun/cleaned_data/eval_balanced/',
        'The dirname with validation data.')

    flags.DEFINE_integer(
        'sampling_rate', 16000, 'samplingrate on audio data')

    flags.DEFINE_float(
        'time_segment_length', 1.0, 'time_segment_length for featuring on audio data')

    flags.DEFINE_float(
        'time_step', 0.5, 'time stepping of features audio data')

    def main(argv):

        # data parser:
        flags_dict = flags.FLAGS.flag_values_dict()
        feature_type = flags_dict['feature_type']
        train_dir = flags_dict['train_dir']
        valid_dir = flags_dict['valid_dir']
        sampling_rate = flags_dict['sampling_rate']
        time_segment=flags_dict['time_segment_length']
        time_step=flags_dict['time_step']

        # train set features extraction and save
        train_set_extractor = AudioFeatureExtraction(train_dir, sampling_rate)

        if feature_type == 'signal':
            train_set_extractor.extract_signal_segments(time_segment, time_step)
        elif feature_type == 'cnn':
            train_set_extractor.extract_cnn_features()
            train_set_extractor.balance_features(True)
        elif feature_type == 'mlp':
            train_set_extractor.extract_mlp_features()
            train_set_extractor.balance_features(True)
        else:
            raise ValueError(
                'model_type not properly defined, only support signal, cnn or mlp')

        train_set_extractor.save_features(feature_type, train_dir)

        # validation set features extraction and save
        validation_set_extractor = AudioFeatureExtraction(valid_dir, sampling_rate)

        if feature_type == 'signal':
            validation_set_extractor.extract_signal_segments(time_segment, time_step)
        elif feature_type == 'cnn':
            validation_set_extractor.extract_cnn_features()
        elif feature_type == 'mlp':
            validation_set_extractor.extract_mlp_features()
        else:
            raise ValueError(
                'model_type not properly defined, only support signal, cnn or mlp')

        validation_set_extractor.save_features(feature_type, valid_dir)

    from absl import app
    app.run(main)
