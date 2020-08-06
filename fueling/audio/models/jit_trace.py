#!/usr/bin/env python

import argparse

import torch

from fueling.learning.train_utils import cuda
from fueling.audio.models.siren_net import SirenNet


def jit_trace_audio_model(torch_model_file, jit_model_file, device):
    model = SirenNet()
    model.load_state_dict(torch.load(torch_model_file))
    model.eval()
    X = torch.ones([1, 1, 33075])
    _ = model.forward(X)
    if device == 'cpu':
        traced_model = torch.jit.trace(model.cpu(), (X,))
    else:
        traced_model = torch.jit.trace(model.cuda(), (cuda(X),))
    traced_model.save(jit_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('torch_model_file', type=str, help='torch model file')
    parser.add_argument('jit_model_file', type=str, help='jit model file')
    parser.add_argument('device', type=str, help='cpu or cuda')
    args = parser.parse_args()
    jit_trace_audio_model(args.torch_model_file, args.jit_model_file, args.device)
