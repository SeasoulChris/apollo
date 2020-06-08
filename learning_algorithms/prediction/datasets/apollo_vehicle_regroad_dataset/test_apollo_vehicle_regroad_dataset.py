#!/usr/bin/env python

import cv2 as cv
import numpy as np

from torch.utils.data import Dataset, DataLoader

from apollo_vehicle_regroad_dataset import ApolloVehicleRegularRoadDataset as ApolloVehicleRegularRoadDataset
from apollo_vehicle_regroad_dataset import collate_fn as collate_fn


def point_to_idx(point_x, point_y):
    return (int((point_x + 40) / 0.1), int((point_y + 40) / 0.1))


def plot_img(obs_features, lane_features, labels, count):
    # black background
    img = np.zeros([1000, 800, 3], dtype=np.uint8)

    # draw boundaries
    cv.circle(img, (400, 400), 2, color=[255, 255, 255], thickness=4)
    cv.line(img, (0, 0), (799, 0), color=[255, 255, 255])
    cv.line(img, (799, 0), (799, 999), color=[255, 255, 255])
    cv.line(img, (0, 999), (0, 0), color=[255, 255, 255])
    cv.line(img, (799, 999), (0, 999), color=[255, 255, 255])
    num_lane_seq = lane_features.shape[0]

    for obs_hist_pt in range(20):
        cv.circle(img, point_to_idx(
            obs_features[2 + obs_hist_pt * 9], obs_features[1 + obs_hist_pt * 9]), radius=3, color=[128, 128, 128])

    for lane_idx in range(num_lane_seq):
        curr_lane = lane_features[lane_idx]
        color_to_use = [0, 128, 128]
        if labels[lane_idx] == 1:
            continue
        for point_idx in range(149):
            cv.line(img, point_to_idx(curr_lane[point_idx * 4], curr_lane[point_idx * 4 + 1]),
                    point_to_idx(curr_lane[point_idx * 4 + 4], curr_lane[point_idx * 4 + 5]),
                    color=color_to_use)

    for lane_idx in range(num_lane_seq):
        curr_lane = lane_features[lane_idx]
        color_to_use = [255, 0, 0]
        if labels[lane_idx] == 0:
            continue
        for point_idx in range(149):
            cv.line(img, point_to_idx(curr_lane[point_idx * 4], curr_lane[point_idx * 4 + 1]),
                    point_to_idx(curr_lane[point_idx * 4 + 4], curr_lane[point_idx * 4 + 5]),
                    color=color_to_use)

    cv.imwrite('img={}__laneseq={}.png'.format(count, num_lane_seq), cv.flip(cv.flip(img, 0), 1))


if __name__ == '__main__':
    dataset_path = '/home/jiacheng/work/apollo/data/apollo_vehicle_regroad_data/test_data_preprocessing/train_data/lane_changes'
    test_dataset = ApolloVehicleRegularRoadDataset(
        dataset_path, is_lane_scanning=True, training_mode=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=1, collate_fn=collate_fn)

    count = 0
    for i, (X, y) in enumerate(test_dataloader):
        if count == 200:
            break
        if (y[1][0, 0] == 1):
            plot_img(X[1].numpy().reshape(-1), X[3].numpy(), y[0].numpy().reshape(-1), count)
            count += 1
