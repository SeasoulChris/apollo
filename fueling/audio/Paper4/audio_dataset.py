#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset

from fueling.audio.Paper3.pyAudioAnalysis import audioBasicIO
from fueling.audio.Paper3.pyAudioAnalysis import audioFeatureExtraction


class AudioDataset(Dataset):
    def __init__(self, data_dir, mode='cnn1d'):
        # TODO(kechxu): load data from .wav files under data_dir
        pass

    def __len__(self):
        # TODO(kechxu): return the size of the dataset
        pass

    def __getitem__(self, idx):
        # TODO(kechxu): return the data at idx
        pass


if __name__ == "__main__":
    print("hello")
