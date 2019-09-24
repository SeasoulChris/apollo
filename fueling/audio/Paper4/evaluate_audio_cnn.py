#!/usr/bin/env python
import os

from absl import flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from audio_cnn import AudioCNN1dModel, AudioCNN2dModel

flags.DEFINE_string(
    'model_file',
    '/home/xukecheng/work/apollo-fuel/model_epoch7_valloss0.793750.pt',
    'model file for evaluation.')
flags.DEFINE_string(
    'valid_dir', '/home/xukecheng/Desktop/cleaned_data/eval_balanced/',
    'dir containing validation data.')


def evaluate(model, X, y_true):
    model.eval()
    y_pred = model(X)
    tag_pred = (y_pred > 0.5)
    tag_true = (y_true > 0.5)
    tag_pred = tag_pred.view(-1)
    tag_true = tag_true.view(-1)
    accuracy = (tag_pred == tag_true).type(torch.float).mean()
    return accuracy


if __name__ == '__main__':

    def main(argv):

        flags_dict = flags.FLAGS.flag_values_dict()

        MODEL = 'cnn2d'

        model = None
        if MODEL == 'cnn1d':
            model = AudioCNN1dModel()
        elif MODEL == 'cnn2d':
            model = AudioCNN2dModel()

        model_state_dict = torch.load(
            flags_dict['model_file'])
        model.load_state_dict(model_state_dict)
        print(model)
        X = np.load(
            os.path.join(flags_dict['valid_dir'], 'features.npy'))
        y = np.load(os.path.join(flags_dict['valid_dir'], 'labels.npy'))

        X_em = X[y == 1]
        y_em = y[y == 1]
        X_nonem = X[y == 0]
        y_nonem = y[y == 0]

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        X_em = torch.from_numpy(X_em)
        y_em = torch.from_numpy(y_em)
        X_nonem = torch.from_numpy(X_nonem)
        y_nonem = torch.from_numpy(y_nonem)

        if MODEL == 'cnn2d':
            X = X.view(-1, 1, 128, 16)
            X_em = X_em.view(-1, 1, 128, 16)
            X_nonem = X_nonem.view(-1, 1, 128, 16)

        print('----------- Data level Results -----------')
        print('Overall accuracy = {}'.format(evaluate(model, X, y)))
        print('EM accuracy = {}'.format(evaluate(model, X_em, y_em)))
        print('Non-EM accuracy = {}'.format(evaluate(model, X_nonem, y_nonem)))

        print('----------- File level Results -----------')
        # TODO(kechxu): implement

    from absl import app
    app.run(main)
