#!/usr/bin/env python

import argparse
import os

import torch

import fueling.common.logging as logging

from fueling.learning.train_utils import train_valid_dataloader
from fueling.planning.models.semantic_map_model import SemanticMapModel

if __name__ == "__main__":
    state_dict_path = '/fuel/model_epoch1_valloss87828.465469.pt'
    logging.info('loading model :{}'.format(state_dict_path))

    model = SemanticMapModel(80, 20)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    X = (torch.ones([2, 3, 224, 224]),
         torch.ones([2, 14]),
         )
    y = model.forward(X)
    traced_cpu_model = torch.jit.trace(model, (X, ))
    traced_cpu_model.save("/fuel/model_epoch1_valloss87828.465469.traced.pt")
