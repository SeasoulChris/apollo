#!/usr/bin/env python

import math

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from semantic_map_single_image_model import *


if __name__ == '__main__':

    model = TrajectoryPredictionSingle(10, 20)
    model_path = "model_epoch2_valloss7.153702.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # construct input
    X = []
    img = torch.randn(3, 960, 960)
    pos = torch.randn(1, 7, 2)
    rois = np.array([[[657.3994, 887.6397],
                      [971.9477, 640.5378],
                      [1219.0497, 955.0861],
                      [904.50134, 1202.188]]])
    rois_tensor = torch.from_numpy(rois)
    scale_factor = torch.tensor(0.47999998927116394)
    X.append((img, pos, pos, rois_tensor, scale_factor))

    # Export the model
    torch.onnx.export(model,
                      X,
                      "final.onnx",
                      export_params=True,
                      opset_version=9,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['pred_traj_tensor'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'pred_traj_tensor': {0: 'batch_size'}}, verbose=True)
