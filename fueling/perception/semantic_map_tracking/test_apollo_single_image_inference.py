#!/usr/bin/env python

import cv2 as cv
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from semantic_map_single_image_model import *
from semantic_map_single_image_dataset import *


#########################################################
# Helper functions
#########################################################
def cuda(x):
    if isinstance(x, list):
        return [cuda(y) for y in x]
    if isinstance(x, tuple):
        return tuple([cuda(y) for y in x])
    return x.cuda() if torch.cuda.is_available() else x


# assuming the input is numobs x 10 x 2
def plot_img_semantic_map(idx, img, labels, pred, ref_pos):
    print('labels size {} and pred size {} and image shape {}'.format(
        labels.shape, pred.shape, img.shape))
    numobs = labels.shape[0]

    for i in range(numobs):
        label_single = labels[i] + ref_pos[i][:2]
        pred_single = pred[i] + ref_pos[i][:2]
        for j in range(10):
            cv.circle(img, (label_single[j][0], label_single[j][1]), radius=3, color=[255, 0, 0])
            cv.circle(img, (pred_single[j][0], pred_single[j][1]), radius=3,
                      color=[0, 0, 255], thickness=-1)
    cv.imwrite("/tmp/test_{}.jpg".format(idx), img)


if __name__ == '__main__':
    # Unit test without image
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataset_path = 'fueling/perception/semantic_map_tracking/test/test_data/'
    test_dataset = ApolloSinglePredictionTrajectoryDataset(dataset_path, True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=1, collate_fn=custom_collate)
    count = 0

    for i, (X, y) in enumerate(test_dataloader):
        # print ('=======================')
        model = TrajectoryPredictionSingle(10, 20)
        model_path = "fueling/perception/semantic_map_tracking/test/model_epoch2_valloss7.153702.pt"
        model.load_state_dict(torch.load(model_path))
        model.cuda().eval()
        pred = model.forward(cuda(X))
        ref_pos = X[0][2].cpu().detach().numpy()
        img = X[0][5].cpu().numpy()
        labels = torch.cat(y).numpy()
        pred = pred.cpu().detach().numpy()
        plot_img_semantic_map(count, img, labels, pred, ref_pos)
        count += 1
