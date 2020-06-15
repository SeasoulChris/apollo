#!/usr/bin/env python

import argparse
import os

import torch

from fueling.planning.models.trajectory_imitation_model import \
    TrajectoryImitationCNNModel, \
    TrajectoryImitationRNNModel, \
    TrajectoryImitationRNNMoreConvModel, \
    TrajectoryImitationRNNUnetResnet18Modelv1, \
    TrajectoryImitationRNNUnetResnet18Modelv2, \
    TrajectoryImitationCNNFCLSTM

import fueling.common.logging as logging
from fueling.learning.train_utils import cuda


def export_torchscript(torch_model_file, jit_model_file, device):
    model = TrajectoryImitationRNNModel([200, 200])
    model_state_dict = torch.load(torch_model_file)
    model.load_state_dict(model_state_dict)
    model.eval()
    X = (torch.ones([1, 12, 200, 200]), torch.ones(
        [1, 1, 200, 200]), torch.ones([1, 1, 200, 200]))
    traced_model = None
    if device == 'gpu':
        traced_model = torch.jit.trace(model.cuda(), (cuda(X),))
    else:
        traced_model = torch.jit.trace(model.cpu(), (X,))

    traced_model.save(jit_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('torch_model_file', type=str, help='torch model file')
    parser.add_argument('jit_model_file', type=str, help='jit model file')
    parser.add_argument('device', type=str, help='cpu or gpu')
    args = parser.parse_args()
    export_torchscript(args.torch_model_file,
                       args.jit_model_file,
                       args.device)
