#!/usr/bin/env python
import os
import warnings
import sys

from absl import flags
import librosa
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from scipy.signal import butter, lfilter, hilbert
from sklearn.utils import shuffle

from fueling.audio.models.audio_features_extraction import AudioFeatureExtraction
from fueling.audio.models.audio_torch_models import AudioCNN1dModel, AudioCNN2dModel, AudioMLPModel
from fueling.audio.models.audio_features_extraction import preprocess
from fueling.audio.pyAudioAnalysis import audioFeatureExtraction
from fueling.common import file_utils


sns.set_style("whitegrid")
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def evaluate(model, X, y_true):
    model.eval()
    y_pred = model(X)
    tag_pred = (y_pred > 0.5)
    tag_true = (y_true > 0.5)
    tag_pred = tag_pred.view(-1)
    tag_true = tag_true.view(-1)
    accuracy = (tag_pred == tag_true).type(torch.float).mean()
    return accuracy


def file_correctly_predicted(model, file_path, model_type,
                             win_size=16, step=8, N=1, threshold=0.5):
    '''
    @return true or false if the file is correctly predicted and detailed results
    '''
    model.eval()
    label = 1
    if file_path.find("nonEmergency") != -1:
        label = 0

    signal, sr = librosa.load(file_path, sr=8000)
    signal = preprocess(signal)
    features = []
    if model_type != 'mlp':
        S = librosa.feature.melspectrogram(signal, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        start = 0
        while start + win_size <= log_S.shape[1]:
            end = start + win_size
            log_S_segment = log_S[:, start: end]
            features.append(log_S_segment)
            start += step
    else:
        features = audioFeatureExtraction.stFeatureExtraction(
            signal, sr, 0.10 * sr, .05 * sr)

    X = np.array(features)
    X = torch.from_numpy(X).float()
    if model_type == 'cnn2d':
        X = X = X.view(-1, 1, 128, win_size)

    y_pred = model(X)
    y_pred = y_pred.view(-1)
    y_pred = y_pred.detach().numpy()

    class_list = []
    label_list = []
    for i in range(y_pred.shape[0] - N):
        avg = np.mean(y_pred[i: i + N])
        if avg > threshold:
            class_list.append(1)
        else:
            class_list.append(0)
        label_list.append(label)

    class_list = np.array(class_list)
    label_list = np.array(label_list)
    results = (class_list == label_list)
    correct = np.mean(results) > threshold

    return correct, results


if __name__ == '__main__':

    flags.DEFINE_string(
        'model_file',
        '/home/jinyun/Dev/apollo-fuel/fueling/audio/models/model_mlp.pt',
        'model file for evaluation.')

    flags.DEFINE_string(
        'model_type', 'mlp', 'Model type, [cnn1d, cnn2d, mlp].')

    flags.DEFINE_string(
        'valid_dir', '/home/jinyun/cleaned_data/eval_balanced/',
        'dir containing validation data.')

    def main(argv):

        flags_dict = flags.FLAGS.flag_values_dict()

        model_type = flags_dict['model_type']

        valid_dir = flags_dict['valid_dir']

        model = None
        feature_type = None
        if model_type == 'cnn1d':
            model = AudioCNN1dModel()
            feature_type = 'cnn'
        elif model_type == 'cnn2d':
            model = AudioCNN2dModel()
            feature_type = 'cnn'
        elif model_type == 'mlp':
            model = AudioMLPModel()
            feature_type = 'mlp'

        model_state_dict = torch.load(
            flags_dict['model_file'])
        model.load_state_dict(model_state_dict)
        print(model)

        X, y = AudioFeatureExtraction.load_features_labels(
            feature_type, valid_dir)

        X_em = X[y == 1]
        y_em = y[y == 1]
        X_nonem = X[y == 0]
        y_nonem = y[y == 0]

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        X_em = torch.from_numpy(X_em).float()
        y_em = torch.from_numpy(y_em).float()
        X_nonem = torch.from_numpy(X_nonem).float()
        y_nonem = torch.from_numpy(y_nonem).float()

        if model_type == 'cnn2d':
            X = X.view(-1, 1, 128, 16)
            X_em = X_em.view(-1, 1, 128, 16)
            X_nonem = X_nonem.view(-1, 1, 128, 16)

        print('----------- Data level Results -----------')
        print('Overall accuracy = {}'.format(evaluate(model, X, y)))
        print('EM accuracy = {}'.format(evaluate(model, X_em, y_em)))
        print('Non-EM accuracy = {}'.format(evaluate(model, X_nonem, y_nonem)))

        # print('----------- File level Results -----------')
        # test_path_em = os.path.join(flags_dict['valid_dir'], 'Emergency/')
        # test_em_files = file_utils.list_files(test_path_em)
        # print('Evaluating test em files ...')
        # em_total = 0
        # em_correct = 0
        # for test_file in tqdm(test_em_files):
        #     if test_file.find('.wav') == -1:
        #         continue
        #     correct, results = file_correctly_predicted(
        #         model, test_file, model_type)
        #     if results.shape[0] == 0:
        #         continue
        #     em_total += 1
        #     if correct:
        #         em_correct += 1
        # print("Correct EM count = {}".format(em_correct))
        # print("Total EM count = {}".format(em_total))
        # print("File level EM accuracy = {}".format(em_correct / em_total))

        # test_path_nonem = os.path.join(
        #     flags_dict['valid_dir'], 'nonEmergency/')
        # test_nonem_files = file_utils.list_files(test_path_nonem)
        # print('Evaluating test nonem files ...')
        # nonem_total = 0
        # nonem_correct = 0
        # for test_file in tqdm(test_nonem_files):
        #     if test_file.find('.wav') == -1:
        #         continue
        #     correct, results = file_correctly_predicted(
        #         model, test_file, model_type)
        #     if results.shape[0] == 0:
        #         continue
        #     nonem_total += 1
        #     if correct:
        #         nonem_correct += 1
        # print("Correct Non-EM count = {}".format(nonem_correct))
        # print("Total Non-EM count = {}".format(nonem_total))
        # print("File level Non-EM accuracy = {}".format(nonem_correct / nonem_total))

    from absl import app
    app.run(main)
