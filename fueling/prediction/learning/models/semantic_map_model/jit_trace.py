#!/usr/bin/env python

import argparse

import torch

from fueling.prediction.learning.models.semantic_map_model.semantic_map_model \
    import SemanticMapSelfLSTMModel


def jit_trace_semantic_map_model(torch_model_file, jit_model_file, device):
    # TODO(kechxu) check input organization in the model definition
    model = SemanticMapSelfLSTMModel(30, 20)
    model.load_state_dict(torch.load(torch_model_file))
    model.eval()
    X = (torch.ones([1, 3, 224, 224]), torch.ones([1, 20, 2]), torch.ones([1, 20, 2]))
    _ = model.forward(X)
    if device == 'cpu':
        traced_model = torch.jit.trace(model.cpu(), (X,))
    else:
        traced_model = torch.jit.trace(model.cuda(), (X.cuda(),))
    traced_model.save(jit_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('torch_model_file', type=str, help='torch model file')
    parser.add_argument('jit_model_file', type=str, help='jit model file')
    parser.add_argument('device', type=str, help='cpu or cuda')
    args = parser.parse_args()
    jit_trace_semantic_map_model(args.torch_model_file, args.jit_model_file, args.device)
