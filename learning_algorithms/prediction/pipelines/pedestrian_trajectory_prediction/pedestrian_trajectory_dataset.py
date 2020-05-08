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
from learning_algorithms.prediction.data_preprocessing.map_feature.online_mapping import ObstacleMapping

MAX_OBS_HISTORY_SIZE = 20
OFFSET_X = semantic_map_config['offset_x']
OFFSET_Y = semantic_map_config['offset_y']

'''
[scene, scene, ..., scene]
  scene: [data_pt, data_pt, ..., data_pt]
    data_pt: [history, future, id]
      history: [feature, feature, ..., feature]
        feature: [timestamp, x, y, heading, polygon_points]
          polygon_points: x, y, x, y, ..., x, y
      future: [x, y, x, y, ..., x, y]
      id: [id]
'''
class PedestrianTrajectoryDataset(Dataset):
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

        self.data_pt_list = []

        all_file_paths = file_utils.list_files(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path, allow_pickle=True).tolist()

            for scene_id, scene in enumerate(file_content):
                if self.SceneHasInvalidDataPt(scene):
                    continue
                for data_pt_id, data_pt in enumerate(scene):
                    if len(data_pt[0]) == 0 or len(data_pt[1]) == 0:
                        continue
                    self.data_pt_list.append((file_path, scene_id, data_pt_id))

        logging.info('Total number of data points = {}'.format(len(self.data_pt_list)))
                    
    def __len__(self):
        return len(self.data_pt_list)

    def __getitem__(self, idx):
        file_path, scene_id, data_pt_id = self.data_pt_list[idx]
        # Get map_region from file_path
        kRegions = ["baidudasha", "XiaMen", "XiongAn"]
        map_region = next((region for region in kRegions if region in file_path), "unknown")

        # read file_content from the file_path
        file_content = np.load(file_path, allow_pickle=True).tolist()
        scene = file_content[scene_id]
        obs_polygons = []
        # Target obstacle's historical information
        world_coord = [0.0, 0.0, 0.0]
        target_obs_hist_size = np.zeros([1,1])
        target_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
        target_obs_pos_abs = np.zeros([MAX_OBS_HISTORY_SIZE, 2])
        target_obs_pos_rel = np.zeros_like(target_obs_pos_abs)
        target_obs_pos_step = np.zeros_like(target_obs_pos_abs)
        target_obs_future_traj_rel = np.zeros([self.pred_len, 2])

        ego_pos_abs = np.zeros([MAX_OBS_HISTORY_SIZE, 2])
        ego_pos_rel = np.zeros_like(ego_pos_abs)
        ego_pos_step = np.zeros_like(ego_pos_abs)
        ego_history = []

        for id, data_pt in enumerate(scene):
            if data_pt[2][0] == -1:
                continue
            # it's target predicting obs
            curr_history = np.array(data_pt[0])
            curr_future = np.array(data_pt[1])
            if self.shifted:
                curr_history = self.RecoverHistory(curr_history)
                curr_future = self.RecoverFuture(curr_future)
            # get curr_obs_polygons and append to obs_polygons
            curr_obs_hist_size = curr_history.shape[0]
            curr_hist_start = MAX_OBS_HISTORY_SIZE - curr_obs_hist_size
            curr_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
            curr_obs_polygons[:, curr_hist_start:, :, :] = \
                (curr_history[:, 4:]).reshape([1, curr_obs_hist_size, 20, 2])
            obs_polygons.append(curr_obs_polygons)
            if id == data_pt_id:
                target_obs_hist_size = np.ones([1, 1]) * curr_obs_hist_size
                target_obs_polygons = curr_obs_polygons[0]
                target_obs_pos_abs[curr_hist_start:, :] = curr_history[:, 1:3]
                world_coord = list(curr_history[-1, 1:4])
                if curr_obs_hist_size > 1:
                    diff_x = curr_history[-1, 1] - curr_history[-2, 1]
                    diff_y = curr_history[-1, 2] - curr_history[-2, 2]
                    world_coord[-1] = math.atan2(diff_y, diff_x)
                # go over history and fill target_obs_pos_rel and step
                for i in range(curr_hist_start, 20):
                    target_obs_pos_rel[i, :] = \
                        CoordUtils.world_to_relative(target_obs_pos_abs[i, :], world_coord)
                    if i > 0:
                        target_obs_pos_step[i, :] = target_obs_pos_rel[i, :] - \
                                                    target_obs_pos_rel[i-1, :]
                # go over future trajectory
                curr_future_traj = curr_future.reshape([self.pred_len, 2])
                for i in range(self.pred_len):
                    new_coord = CoordUtils.world_to_relative(curr_future_traj[i, :], world_coord)
                    target_obs_future_traj_rel[i, :] = new_coord

        for id, data_pt in enumerate(scene):
            if data_pt[2][0] != -1:  # This is ego vehicle
                continue
            curr_history = np.array(data_pt[0])
            curr_future = np.array(data_pt[1])
            if self.shifted:
                curr_history = self.RecoverHistory(curr_history)
                curr_future = self.RecoverFuture(curr_future)
            # get curr_obs_polygons and append to obs_polygons
            curr_obs_hist_size = curr_history.shape[0]
            curr_hist_start = MAX_OBS_HISTORY_SIZE - curr_obs_hist_size
            # curr_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
            # curr_obs_polygons[:, curr_hist_start:, :, :] = \
            #     (curr_history[:, 4:]).reshape([1, curr_obs_hist_size, 20, 2])
            # obs_polygons.append(curr_obs_polygons)

            ego_hist_size = np.ones([1, 1]) * curr_obs_hist_size
            # ego_polygons = curr_obs_polygons[0]
            ego_pos_abs[curr_hist_start:, :] = curr_history[:, 1:3]
            ego_history = curr_history[:, 1:4]
            # go over history and fill target_obs_pos_rel and step
            for i in range(curr_hist_start, 20):
                ego_pos_rel[i, :] = CoordUtils.world_to_relative(ego_pos_abs[i, :], world_coord)
                if i > 0:
                    ego_pos_step[i, :] = ego_pos_rel[i, :] - ego_pos_rel[i-1, :]

        obs_polygons = np.concatenate(obs_polygons)
        obs_mapping = ObstacleMapping(map_region, self.base_map[map_region],
                                      world_coord, obs_polygons, ego_history)
        img = obs_mapping.crop_by_history(target_obs_polygons)
        cv.imwrite('/fuel/hehe.png', img)
        origin_img = img.copy()
        if self.img_transform:
            img = self.img_transform(img)

        return ((img,
                 torch.from_numpy(target_obs_pos_abs).float(),
                 torch.from_numpy(target_obs_hist_size).float(),
                 torch.from_numpy(target_obs_pos_rel).float(),
                 torch.from_numpy(target_obs_pos_step).float(),
                 origin_img,
                 torch.from_numpy(ego_pos_rel).float(),
                 torch.from_numpy(ego_pos_step).float()),
                torch.from_numpy(target_obs_future_traj_rel).float())

    # Only for test purpose
    def getitem(self, idx):
        return self.__getitem__(idx)


if __name__ == '__main__':
    pedestrian_dataset = PedestrianTrajectoryDataset('/fuel/kinglong_data/train_clean/')
    pedestrian_dataset.getitem(100)

