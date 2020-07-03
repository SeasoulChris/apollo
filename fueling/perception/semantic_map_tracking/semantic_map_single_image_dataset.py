#!/usr/bin/env python

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

from fueling.common.coord_utils import CoordUtils
import fueling.common.file_utils as file_utils
from fueling.learning.network_utils import *
from fueling.prediction.learning.data_preprocessing.map_feature.online_mapping \
    import ObstacleMapping

obs_hist_size = 20
obs_unit_feature_size = 40 + 9
obs_feature_size = obs_hist_size * obs_unit_feature_size
single_lane_feature_size = 600
past_lane_feature_size = 200
future_lane_feature_size = 400


class ApolloSinglePredictionTrajectoryDataset(Dataset):
    def __init__(self, data_dir, img_mode=True,
                 pred_len=30, basedir='fueling/perception/semantic_map_tracking/test/'):
        self.pred_len = pred_len
        self.img_mode = img_mode
        self.map_region = []
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.obs_hist_sizes = []
        self.obs_pos = []
        self.obs_pos_rel = []
        self.obs_heading = []
        self.obs_polygons = []

        self.lane_feature = []

        self.future_traj = []
        self.future_traj_rel = []
        self.future_heading = []

        self.is_predictable = []

        self.start_idx = []
        self.end_idx = []

        self.reference_world_coord = []
        self.same_scene_mask = []
        self.random_rotate = True
        self.using_image_regression = True
        total_num_cutin_data_pt = 0
        accumulated_data_pt = 0

        # TODO(Hongyi): add the drawing class here.
        self.base_map = {"sunnyvale": cv.imread(
            basedir + "sunnyvale_with_two_offices.png"),
            "san_mateo": cv.imread(basedir + "san_mateo.png")}

        scene_id = -1
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

                    # Get number of lane-sequences info.
                    curr_num_lane_sequence = int(data_pt[0])

                    # Get the size of obstacle state history.
                    curr_obs_hist_size = int(
                        np.sum(np.array(data_pt[1:obs_feature_size + 1:obs_unit_feature_size])))
                    if curr_obs_hist_size <= 0:
                        accumulated_data_pt -= 1
                        continue
                    self.obs_hist_sizes.append(curr_obs_hist_size * np.ones((1, 1)))
                    self.same_scene_mask.append(scene_id * np.ones((1, 1)))

                    # Get map_region
                    if file_path.find("sunnyvale") != -1:
                        self.map_region.append("sunnyvale")
                    elif file_path.find("san_mateo") != -1:
                        self.map_region.append("san_mateo")
                    else:
                        self.map_region.append("unknown")

                    # Get the obstacle position features (organized from past to present).
                    # (if length not enough, then pad heading zeros)
                    curr_obs_feature = np.array(
                        data_pt[1:obs_feature_size + 1]).reshape(
                        (obs_hist_size, obs_unit_feature_size))
                    curr_obs_feature = np.flip(curr_obs_feature, 0)
                    curr_obs_pos = np.zeros((1, obs_hist_size, 2))
                    # (1 x max_obs_hist_size x 2)
                    curr_obs_pos[0, -curr_obs_hist_size:, :] = \
                        curr_obs_feature[-curr_obs_hist_size:, 1:3]
                    self.obs_pos.append(curr_obs_pos)
                    curr_obs_pos_rel = np.zeros((1, obs_hist_size, 2))
                    if curr_obs_hist_size > 1:
                        curr_obs_pos_rel[0, -curr_obs_hist_size + 1:, :] = \
                            curr_obs_pos[0, -curr_obs_hist_size + 1:, :] - \
                            curr_obs_pos[0, -curr_obs_hist_size:-1, :]
                    self.obs_pos_rel.append(curr_obs_pos_rel)

                    # Get the obstacle polygon features (organized from past to present).
                    curr_obs_polygon = curr_obs_feature[:, -40:].reshape((1, obs_hist_size, 20, 2))
                    self.obs_polygons.append(curr_obs_polygon)

                    # Get the lane features.
                    # (curr_num_lane_sequence x num_lane_pts x 4)
                    curr_lane_feature = np.array(
                        data_pt[obs_feature_size
                                + 1:obs_feature_size
                                + 1
                                + (single_lane_feature_size)
                                * curr_num_lane_sequence]).reshape((curr_num_lane_sequence,
                                                                    int(single_lane_feature_size
                                                                        / 4),
                                                                    4))
                    curr_lane_feature[:, :, [0, 1]] = curr_lane_feature[:, :, [1, 0]]
                    # Remove too close lane-points.
                    curr_lane_feature = np.concatenate(
                        (curr_lane_feature[:, :49, :], curr_lane_feature[:, 51:, :]), axis=1)
                    # The following part appends a beginning and an ending point.
                    begin_pt = 2 * curr_lane_feature[:, 0, :] - 1 * curr_lane_feature[:, 1, :]
                    begin_pt[:, 2] = curr_lane_feature[:, 0, 2]
                    begin_pt[:, 3] = np.zeros((curr_num_lane_sequence))
                    end_pt = 2 * curr_lane_feature[:, -1, :] - 1 * curr_lane_feature[:, -2, :]
                    end_pt[:, 2] = curr_lane_feature[:, -1, 2]
                    end_pt[:, 3] = np.zeros((curr_num_lane_sequence))
                    curr_lane_feature = np.concatenate(
                        (begin_pt.reshape((curr_num_lane_sequence, 1, 4)),
                         curr_lane_feature, end_pt.reshape((
                             curr_num_lane_sequence, 1, 4))), axis=1)
                    self.lane_feature.append(curr_lane_feature)

                    # Skip getting label data for those without labels at all.
                    if len(data_pt) <= obs_feature_size + 1 + \
                            (single_lane_feature_size) * curr_num_lane_sequence:
                        self.is_predictable.append(np.zeros((1, 1)))
                        self.reference_world_coord.append([0.0, 0.0, 0.0])
                        self.future_traj.append(np.zeros((1, self.pred_len, 2)))
                        self.future_traj_rel.append(np.zeros((1, self.pred_len - 1, 2)))
                        self.future_heading.append(np.zeros((1, self.pred_len, 1)))
                        continue
                    self.is_predictable.append(np.ones((1, 1)))

                    # Get the future trajectory label.
                    curr_future_traj = np.array(data_pt[-90:-30]).reshape((2, 30))
                    curr_heading = np.array(data_pt[-30:]).reshape((1, 30))
                    curr_future_traj = curr_future_traj[:, :self.pred_len]
                    curr_future_traj = curr_future_traj.transpose()
                    curr_heading = curr_heading[:, :self.pred_len].transpose()
                    # x, y and heading information
                    ref_world_coord = [curr_obs_feature[-1, 1],
                                       curr_obs_feature[-1, 2], curr_obs_feature[-1, 7]]
                    self.reference_world_coord.append(ref_world_coord)
                    new_curr_future_traj = np.zeros((1, self.pred_len, 2))
                    new_curr_future_heading = np.zeros((1, self.pred_len, 1))
                    for i in range(self.pred_len):
                        new_coord = CoordUtils.world_to_relative(
                            curr_future_traj[i, :], ref_world_coord)
                        new_curr_future_traj[0, i, 0] = new_coord[0]
                        new_curr_future_traj[0, i, 1] = new_coord[1]
                        new_curr_future_heading[0, i, 0] = curr_heading[i] - curr_obs_feature[-1, 7]

                    # (1 x self.pred_len x 2)
                    self.future_traj.append(curr_future_traj[None, ...])
                    curr_future_traj_rel = np.zeros((1, self.pred_len - 1, 2))
                    curr_future_traj_rel = new_curr_future_traj[:, 1:, :] \
                        - new_curr_future_traj[:, :-1, :]
                    # save heading difference in theta
                    self.future_heading.append(new_curr_future_heading)
                    # (1 x self.pred_len-1 x 2)
                    self.future_traj_rel.append(curr_future_traj_rel)

                self.end_idx.append(accumulated_data_pt)
                if self.end_idx[-1] == self.start_idx[-1]:
                    self.end_idx.pop()
                    self.start_idx.pop()
        self.total_num_data_pt = len(self.start_idx)
        print('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def get_roi(self, world_coord, obs_mapping, random_angle):
        obs_center = tuple(obs_mapping.get_trans_point(world_coord[0:2]))
        heading_angle = world_coord[2] * 180 / np.pi
        M = cv.getRotationMatrix2D(obs_center, heading_angle - 90 + random_angle, 1.0)

        points = np.array([[obs_center[0] - 200, obs_center[1] - 300, 1],
                           [obs_center[0] + 200, obs_center[1] - 300, 1],
                           [obs_center[0] + 200, obs_center[1] + 100, 1],
                           [obs_center[0] - 200, obs_center[1] + 100, 1]])
        rot_points = np.dot(M, points.transpose())
        return rot_points.transpose()

    def filter_removed(self, entry):
        dim = entry.ndim
        if entry.ndim == 2:
            idx = np.all(entry == 0, axis=1)
            return (entry[~idx], idx)
        return (None, None)

    def transform_scale_rotate_coord(self, ori_coord, obs_mapping, rotationM, scale_factor):
        pt = np.array(obs_mapping.get_trans_point(ori_coord)) * scale_factor
        if rotationM is not None:
            return np.dot(rotationM, np.array([pt[0], pt[1], 1]).transpose())
        else:
            return np.array([pt[0], pt[1]])

    def __getitem__(self, idx):
        s_idx = self.start_idx[idx]
        e_idx = self.end_idx[idx]
        obs_hist_sizes = np.concatenate(self.obs_hist_sizes[s_idx:e_idx])
        obs_polygons = np.concatenate(self.obs_polygons[s_idx:e_idx])
        target_obs_future_traj = np.concatenate(self.future_traj[s_idx:e_idx])
        region = self.map_region[s_idx]
        predictable_prob = np.concatenate(self.is_predictable[s_idx:e_idx])
        predictable_prob = predictable_prob.reshape((-1))
        predictable_prob = predictable_prob / np.sum(predictable_prob)
        predicting_idx = np.random.choice(predictable_prob.shape[0], 1, p=predictable_prob)[0]
        world_coord_center = self.reference_world_coord[s_idx + predicting_idx]
        # TODO(Hongyi): modify the following part to include multiple obstacles.
        obs_mapping = ObstacleMapping(
            region,
            self.base_map[region],
            world_coord_center,
            obs_polygons,
            shift=False)
        # middle feature map represented as image

        all_obs_positions = np.concatenate(self.obs_pos[s_idx:e_idx])
        all_obs_pos_rel = np.zeros((all_obs_positions.shape[0], 7, 2))
        target_obs_pos_rel = np.zeros((target_obs_future_traj.shape[0], 10, 2))
        reference_world_coords = np.zeros((target_obs_future_traj.shape[0], 3))
        rois = np.zeros((all_obs_positions.shape[0], 4, 2))

        img = cv.resize(obs_mapping.feature_map, (960, 960))
        angle = 0
        rotationM = None
        scale_factor = 960 / 2000

        if self.random_rotate:
            angle = random.randrange(-180, 180)
            (img, rotationM) = rotate(img, angle)

        for n, idx in zip(range(0, e_idx - s_idx), range(s_idx, e_idx)):
            hist_size = int(obs_hist_sizes[n])
            world_coord = self.reference_world_coord[idx]
            pt_ori = None
            if np.count_nonzero(np.array(world_coord) == 0):
                print('filtered due to world coord 0 {}'.format(world_coord))
                continue

            reference_world_coords[n, :] = world_coord
            if self.using_image_regression:
                pt = self.transform_scale_rotate_coord(
                    world_coord[0:2], obs_mapping, rotationM, scale_factor)
                reference_world_coords[n, :2] = pt
                pt_ori = pt

            for i in range(20 - hist_size, 20, 3):
                if self.using_image_regression:
                    pt = self.transform_scale_rotate_coord(
                        all_obs_positions[n, i, :], obs_mapping, rotationM, scale_factor)
                    if pt_ori is not None:
                        all_obs_pos_rel[n, int(i / 3), :] = pt - pt_ori
                    else:
                        print('pt_ori 1 should not be null')

            for i, j in zip(range(0, 30, 3), range(0, 10)):
                target_obs_pos_rel[n, j, :] = target_obs_future_traj[n, i, :]
                if self.using_image_regression:
                    pt = self.transform_scale_rotate_coord(
                        target_obs_pos_rel[n, j, :], obs_mapping, rotationM, scale_factor)
                    if pt_ori is not None:
                        target_obs_pos_rel[n, j, :] = pt - pt_ori
                    else:
                        print('pt_ori 2 should not be null')

            rois[n, :] = self.get_roi(world_coord, obs_mapping, angle)

        # assert(all_obs_pos_rel.shape[0]==target_obs_pos_rel.shape[0])
        (reference_world_coords, idx) = self.filter_removed(reference_world_coords)
        all_obs_pos_rel = all_obs_pos_rel[~idx]
        rois = rois[~idx]
        target_obs_pos_rel = target_obs_pos_rel[~idx]
        assert(all_obs_pos_rel.shape[0] == target_obs_pos_rel.shape[0])

        # save original image
        ori_img = img

        if self.img_transform:
            img = self.img_transform(img)

        return ((img,
                 torch.from_numpy(all_obs_pos_rel).float(),
                 torch.from_numpy(reference_world_coords).float(),
                 torch.from_numpy(rois).float(),
                 torch.tensor(scale_factor),
                 torch.from_numpy(ori_img),
                 ),
                torch.from_numpy(target_obs_pos_rel).float())


# a simple custom collate function, just to show the idea
def custom_collate(batch):
    # make sure at least one row
    data = [item[0] for item in batch if item[1].shape[0] > 0]
    target = [item[1] for item in batch if item[1].shape[0] > 0]
    return [data, target]
