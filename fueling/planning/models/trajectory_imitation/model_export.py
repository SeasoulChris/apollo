#!/usr/bin/env python

import argparse

import torch

from fueling.learning.train_utils import cuda
from fueling.planning.models.trajectory_imitation.cnn_fc_model import TrajectoryImitationCNNFC
import fueling.common.logging as logging


def export(torch_model_file, export_model_file, device, export_form):
    model = TrajectoryImitationCNNFC()
    # model = TrajectoryImitationSelfCNNLSTM(10, 10)
    # model = TrajectoryImitationSelfCNNLSTMWithRasterizer(10, 10, [200, 200])
    # model = TrajectoryImitationConvRNN([200, 200])
    # model = TrajectoryImitationDeeperConvRNN([200, 200])
    # model = TrajectoryImitationConvRNNUnetResnet18v1([200, 200])
    # model = TrajectoryImitationConvRNNUnetResnet18v2([200, 200])
    model_state_dict = torch.load(torch_model_file)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    X = torch.ones([1, 12, 200, 200])
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 10, 4]), torch.ones([1, 10, 4]))
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 10, 4]), torch.ones([1, 10, 4]))
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 1, 200, 200]), torch.ones([1, 1, 200, 200]))
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 1, 200, 200]), torch.ones([1, 1, 200, 200]))
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 1, 200, 200]), torch.ones([1, 1, 200, 200]))
    # X = (torch.ones([1, 12, 200, 200]), torch.ones(
    # [1, 1, 200, 200]), torch.ones([1, 1, 200, 200]))
    if export_form == 'torchscript':
        traced_model = None
        if device == 'gpu':
            traced_model = torch.jit.trace(model.cuda(), (cuda(X),))
        else:
            traced_model = torch.jit.trace(model.cpu(), (X,))
        traced_model.save(export_model_file)

    elif export_form == 'onnx':
        if device == 'gpu':
            torch.onnx.export(model.cuda(), (cuda(X),),
                              export_model_file, verbose=False)
            # import onnx
            # m = onnx.load(onnx_file)
            # print(m.graph.node)
        else:
            torch.onnx.export(model.cpu(), (X,), export_model_file)
    else:
        logging.error('export_form not supported or mistyped')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('torch_model_file', type=str, help='torch model file')
    parser.add_argument('export_model_file', type=str,
                        help='export model file')
    parser.add_argument('device', type=str, help='cpu or gpu')
    parser.add_argument('export_form', type=str, help='onnx or torchscript')
    args = parser.parse_args()
    export(args.torch_model_file,
           args.export_model_file,
           args.device,
           args.export_form)
