#!/usr/bin/env python

import cv2 as cv
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from fueling.common.coord_utils import CoordUtils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
from fueling.prediction.common.configure import semantic_map_config
from fueling.prediction.learning.data_preprocessing.map_feature.online_mapping \
    import ObstacleMapping

MAX_OBS_HISTORY_SIZE = 20
OFFSET_X = semantic_map_config['offset_x']
OFFSET_Y = semantic_map_config['offset_y']

'''
[scene, scene, ..., scene]
  scene: [obstacle, obstacle, ..., obstacle]
    obstacle: [history, id, cross_road]
      history: [feature, feature, ..., feature]
        feature: [timestamp, x, y, heading, polygon_points]
          polygon_points: x, y, x, y, ..., x, y # 20 (x,y)
      id: [obstacle_id]
      cross_road: [1 or 0]
'''


class PedestrianIntentionDataset(Dataset):
    def RecoverHistory(self, history):
        history[:, 1] += OFFSET_X
        history[:, 2] += OFFSET_Y
        history[:, 4::2] += OFFSET_X
        history[:, 5::2] += OFFSET_Y
        return history

    def RecoverFuture(self, future):
        future[0::2] += OFFSET_X
        future[1::2] += OFFSET_Y
        return future

    def SceneHasInvalidDataPt(self, scene):
        for data_pt in scene:
            if len(data_pt[1]) == 0:
                continue
            if len(data_pt[0]) == 0:
                return True
            curr = data_pt[0][-1]
            curr_x = data_pt[0][-1][1]
            curr_y = data_pt[0][-1][2]
            if self.shifted:
                curr_x += OFFSET_X
                curr_y += OFFSET_Y
            if abs(curr_x) < 1.0 or abs(curr_y) < 1.0:
                return True
        return False

    def __init__(self, data_dir, pred_len=30, shifted=True):
        self.pred_len = pred_len
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.base_map = {"baidudasha": cv.imread("/fuel/testdata/map_feature/baidudasha.png"),
                         "XiaMen": cv.imread("/fuel/testdata/map_feature/XiaMen.png"),
                         "XiongAn": cv.imread("/fuel/testdata/map_feature/XiongAn.png")}
        self.shifted = shifted

        self.scene_list = []

        all_file_paths = file_utils.list_files(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path, allow_pickle=True).tolist()

            for scene_id, scene in enumerate(file_content):
                if self.SceneHasInvalidDataPt(scene):
                    continue
                self.scene_list.append((file_path, scene_id))

        logging.info('Total number of data points = {}'.format(len(self.scene_list)))

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        file_path, scene_id = self.scene_list[idx]
        # Get map_region from file_path
        kRegions = ["baidudasha", "XiaMen", "XiongAn"]
        map_region = next((region for region in kRegions if region in file_path), "unknown")

        # read file_content from the file_path
        file_content = np.load(file_path, allow_pickle=True).tolist()
        scene = file_content[scene_id]

        obs_polygons = []
        cross_polygons = []
        ego_history = []

        for id, data_pt in enumerate(scene):
            if data_pt[1][0] == -1:
                curr_history = np.array(data_pt[0])
                if self.shifted:
                    curr_history = self.RecoverHistory(curr_history)
                ego_history = curr_history[:, 1:4]
            else:
                # it's target predicting obs
                curr_history = np.array(data_pt[0])
                if self.shifted:
                    curr_history = self.RecoverHistory(curr_history)
                # get curr_obs_polygons and append to obs_polygons
                curr_obs_hist_size = curr_history.shape[0]
                curr_hist_start = MAX_OBS_HISTORY_SIZE - curr_obs_hist_size
                curr_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
                curr_obs_polygons[:, curr_hist_start:, :, :] = \
                    (curr_history[:, 4:]).reshape([1, curr_obs_hist_size, 20, 2])
                obs_polygons.append(curr_obs_polygons)
                if data_pt[2][0]:
                    curr_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
                    curr_obs_polygons[0, -1, :, :] = curr_history[0, 4:].reshape([20, 2])
                    cross_polygons.append(curr_obs_polygons)


        obs_polygons = np.concatenate(obs_polygons)
        cross_polygons = np.concatenate(cross_polygons)
        world_coord = ego_history[-1]
        obs_mapping = ObstacleMapping(map_region, self.base_map[map_region],
                                      world_coord, obs_polygons, ego_history)
        feature_img = obs_mapping.crop_ego_center()

        cross_img = ObstacleMapping(map_region, np.zeros_like(self.base_map[map_region]),
                                       world_coord, cross_polygons, ego_history).crop_ego_center(color=(0,0,0))
        cross_img = np.sum(cross_img, axis=2)
        cross_img = cross_img / np.max(cross_img) * 1.0
        # cv.imwrite('/fuel/hehe.png', feature_img)
        origin_img = feature_img.copy()
        if self.img_transform:
            feature_img = self.img_transform(feature_img)

        return ((feature_img, origin_img),
                cross_img)

    # Only for test purpose
    def getitem(self, idx):
        return self.__getitem__(idx)


if __name__ == '__main__':
    pedestrian_dataset = PedestrianIntentionDataset('/fuel/training_data/')
    pedestrian_dataset.getitem(100)
