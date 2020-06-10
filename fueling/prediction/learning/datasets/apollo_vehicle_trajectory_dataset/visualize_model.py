#!/usr/bin/env python

import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms

import fueling.prediction.learning.models.semantic_map_model.semantic_map_model as semantic_map_model
from apollo_vehicle_trajectory_dataset import ApolloVehicleTrajectoryDataset as ApolloVehicleTrajectoryDataset

dataset_path = '/data/training_data/train/sunnyvale/2019-01-03/'
test_dataset = ApolloVehicleTrajectoryDataset(dataset_path, True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
# it = iter(test_dataloader)
# X, y = next(it)

# Should also change model to output img_attn before visualize
model = semantic_map_model.SemanticMapSelfAttentionLSTMModel(30, 20, cnn_net=models.mobilenet_v2)
state_dict_path = '/fuel/model_epoch17_valloss0.789887.state_dict'
model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
model.eval()


for i, (X, traj) in enumerate(test_dataloader):
    origin_img = X[10]
    origin_img = origin_img.view(224, 224, 3).detach().numpy()
    cv.imwrite("/fuel/img{}.png".format(i), origin_img)
    (pred, img_attn) = model.forward(X)
    # print(traj)
    img_attn = img_attn.view(224, 224, 1).detach().numpy()
    img_after_attn = origin_img * img_attn
    cv.imwrite("/fuel/img_after_attn{}.png".format(i), img_after_attn)
    cv.imwrite("/fuel/img_attn{}.png".format(i), img_attn / np.max(img_attn) * 255)
    if i == 1:
        break
