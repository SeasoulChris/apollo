#!/usr/bin/env python

import numpy as np
import os
import torch
from torch.utils.data import Dataset

from fueling.audio.Paper3.pyAudioAnalysis import audioBasicIO
from fueling.audio.Paper3.pyAudioAnalysis import audioFeatureExtraction
from fueling.common import file_utils


class AudioDataset(Dataset):
    def __init__(self, data_dir, mode='cnn1d', win_size=10, step=5):
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
        # TODO(kechxu): return the data at idx
        pass


if __name__ == "__main__":
    print("hello")
