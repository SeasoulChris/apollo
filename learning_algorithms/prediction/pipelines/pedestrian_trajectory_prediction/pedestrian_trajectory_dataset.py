#!/usr/bin/env python

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from fueling.common.coord_utils import CoordUtils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
from learning_algorithms.prediction.data_preprocessing.map_feature.online_mapping import ObstacleMapping

MAX_OBS_HISTORY_SIZE = 20
'''
[scene, scene, ..., scene]
  scene: [data_pt, data_pt, ..., data_pt]
    data_pt: [history, future]
      history: [feature, feature, ..., feature]
        feature: [timestamp, x, y, heading, polygon_points]
          polygon_points: x, y, x, y, ..., x, y
      future: [x, y, x, y, ..., x, y]
'''
class PedestrianTrajectoryDataset(Dataset):
    def __init__(self, data_dir, pred_len=30):
        self.pred_len = pred_len
        self.map_region = []
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.obs_hist_sizes = []
        self.obs_pos = []
        self.obs_pos_rel = []
        self.obs_polygons = []

        self.future_traj = []
        self.future_traj_rel = []

        self.is_predictable = []

        self.start_idx = []
        self.end_idx = []

        self.reference_world_coord = []

        self.base_map = {"baidudasha": cv.imread("/fuel/testdata/map_feature/baidudasha.png")}

        scene_id = -1
        accumulated_data_pt = 0
        all_file_paths = file_utils.list_files(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            file_content = np.load(file_path, allow_pickle=True).tolist()

            for scene in file_content:
                self.start_idx.append(accumulated_data_pt)
                scene_id += 1
                for data_pt in scene:
                    accumulated_data_pt += 1
                    curr_history = np.array(data_pt[0])
                    curr_future = np.array(data_pt[1])
                    curr_obs_hist_size = curr_history.shape[0]
                    if curr_obs_hist_size <= 0:
                        accumulated_data_pt -= 1
                        continue
                    self.obs_hist_sizes.append(curr_obs_hist_size * np.ones((1, 1)))
                    # Get map_region
                    if file_path.find("baidudasha") != -1:
                        self.map_region.append("baidudasha")
                    else:
                        self.map_region.append("unknown")

                    # Get the obstacle position features
                    # (organized from past to present).
                    # (if length not enough, then pad heading zeros)
                    curr_obs_pos = np.zeros((1, MAX_OBS_HISTORY_SIZE, 2))
                    curr_hist_start = MAX_OBS_HISTORY_SIZE - curr_obs_hist_size
                    curr_obs_pos[0, curr_hist_start:, :] = curr_history[:, 1:3]
                    self.obs_pos.append(curr_obs_pos)

                    curr_obs_polygons = np.zeros([1, MAX_OBS_HISTORY_SIZE, 20, 2])
                    curr_obs_polygons[:, curr_hist_start:, :, :] = \
                        (curr_history[:, 4:]).reshape([1, curr_obs_hist_size, 20, 2])
                    self.obs_polygons.append(curr_obs_polygons)

                    curr_obs_pos_rel = np.zeros((1, MAX_OBS_HISTORY_SIZE, 2))
                    if curr_obs_hist_size > 1:
                        curr_obs_pos_rel[0, -curr_obs_hist_size+1:, :] = \
                        curr_obs_pos[0, -curr_obs_hist_size+1:, :] - \
                        curr_obs_pos[0, -curr_obs_hist_size:-1, :]
                    self.obs_pos_rel.append(curr_obs_pos_rel)

                    if len(data_pt[1]) == 0:
                        self.is_predictable.append(np.zeros((1, 1)))
                        self.reference_world_coord.append([0.0, 0.0, 0.0])
                        self.future_traj.append(np.zeros((1, self.pred_len, 2)))
                        self.future_traj_rel.append(np.zeros((1, self.pred_len-1, 2)))
                        continue

                    self.is_predictable.append(np.ones((1, 1)))
                    curr_future_traj = curr_future.reshape([self.pred_len, 2])
                    ref_world_coord = list(curr_history[-1, 1:4])
                    self.reference_world_coord.append(ref_world_coord)
                    new_curr_future_traj = np.zeros((1, self.pred_len, 2))
                    for i in range(self.pred_len):
                        new_coord = CoordUtils.world_to_relative(
                            curr_future_traj[i, :], ref_world_coord)
                        new_curr_future_traj[0, i, 0] = new_coord[0]
                        new_curr_future_traj[0, i, 1] = new_coord[1]
                    # (1 x self.pred_len x 2)
                    self.future_traj.append(new_curr_future_traj)
                    curr_future_traj_rel = np.zeros((1, self.pred_len-1, 2))
                    curr_future_traj_rel = \
                        new_curr_future_traj[:, 1:, :] - new_curr_future_traj[:, :-1, :]
                    # (1 x self.pred_len-1 x 2)
                    self.future_traj_rel.append(curr_future_traj_rel)

                self.end_idx.append(accumulated_data_pt)
                if self.end_idx[-1] == self.start_idx[-1]:
                    self.end_idx.pop()
                    self.start_idx.pop()

        self.total_num_data_pt = len(self.start_idx)
        logging.info('Total number of data points = {}'.format(self.total_num_data_pt))
                    
    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        s_idx = self.start_idx[idx]
        e_idx = self.end_idx[idx]
        obs_hist_sizes = np.concatenate(self.obs_hist_sizes[s_idx:e_idx])
        obs_polygons = np.concatenate(self.obs_polygons[s_idx:e_idx])

        predictable_prob = np.concatenate(self.is_predictable[s_idx:e_idx])
        predictable_prob = predictable_prob.reshape((-1))
        predictable_prob = predictable_prob / (np.sum(predictable_prob))
        predicting_idx = np.random.choice(predictable_prob.shape[0], 1, p=predictable_prob)[0]
        world_coord = self.reference_world_coord[s_idx + predicting_idx]
        target_obs_future_traj = self.future_traj[s_idx + predicting_idx][0, :, :]
        region = self.map_region[s_idx + predicting_idx]
        obs_mapping = ObstacleMapping(region, self.base_map[region], world_coord, obs_polygons)
        img = obs_mapping.crop_by_history(obs_polygons[predicting_idx])
        origin_img = img.copy()
        if self.img_transform:
            img = self.img_transform(img)
        all_obs_positions = np.concatenate(self.obs_pos[s_idx:e_idx])
        all_obs_pos_rel = np.concatenate(self.obs_pos_rel[s_idx:e_idx])

        # Target obstacle's historical information
        target_obs_hist_size = obs_hist_sizes[predicting_idx]
        target_obs_pos_abs = all_obs_positions[predicting_idx, :, :]
        target_obs_pos_rel = np.zeros_like(target_obs_pos_abs)
        target_obs_pos_step = np.zeros_like(target_obs_pos_abs)
        hist_size  = int(target_obs_hist_size[0])
        for i in range(20-hist_size, 20):
            target_obs_pos_rel[i, :] = CoordUtils.world_to_relative(
                target_obs_pos_abs[i, :], world_coord)
            if i > 0:
                target_obs_pos_step[i, :] = target_obs_pos_rel[i, :] - target_obs_pos_rel[i-1, :]

        return ((img,
                 torch.from_numpy(target_obs_pos_abs).float(),
                 torch.from_numpy(target_obs_hist_size).float(),
                 torch.from_numpy(target_obs_pos_rel).float(),
                 torch.from_numpy(target_obs_pos_step).float()),
                torch.from_numpy(target_obs_future_traj).float())

    # Only for test purpose
    def getitem(self, idx):
        return self.__getitem__(idx)


if __name__ == '__main__':
    pedestrian_dataset = PedestrianTrajectoryDataset('/fuel/kinglong_data/train/')
    pedestrian_dataset.getitem(10)
