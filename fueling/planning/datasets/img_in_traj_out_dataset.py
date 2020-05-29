#!/usr/bin/env python

import re
import os
import shutil

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from modules.planning.proto import planning_semantic_map_config_pb2
from modules.planning.proto import learning_data_pb2

from fueling.common.coord_utils import CoordUtils
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
from fueling.planning.datasets.semantic_map_feature.chauffeur_net_feature_generator \
    import ChauffeurNetFeatureGenerator


class TrajectoryImitationCNNDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 input_data_agumentation=False, ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        # TODO(Jinyun): add multi-map support
        # region = "sunnyvale_with_two_offices"

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region, map_path)
        self.input_data_agumentation = input_data_agumentation
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        past_motion_dropout = False
        if self.input_data_agumentation:
            coordinate_heading = np.random.uniform(
                -self.max_rand_coordinate_heading, self.max_rand_coordinate_heading)
            past_motion_dropout = np.random.uniform(0, 1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        past_motion_dropout)
        transformed_img_feature = self.img_transform(img_feature)

        ref_coords = [current_x,
                      current_y,
                      current_theta]
        pred_points = np.zeros((0, 4))
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_v = pred_point.trajectory_point.v
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return (transformed_img_feature,
                    torch.from_numpy(pred_points).float(),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return (transformed_img_feature, torch.from_numpy(pred_points).float())


class TrajectoryImitationRNNDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 input_data_agumentation=False, ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_feature_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.img_bitmap_transform = transforms.Compose([
            # Normalized to [0, 1]
            transforms.ToTensor()])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        # TODO(Jinyun): add multi-map support
        # region = "sunnyvale_with_two_offices"

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(renderer_config_file,
                                         imgs_dir,
                                         region,
                                         map_path,
                                         base_map_update_flag=False)
        self.input_data_agumentation = input_data_agumentation
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        np.random.seed(0)
        self.img_size = [renderer_config.width, renderer_config.height]
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        past_motion_dropout = False
        if self.input_data_agumentation:
            coordinate_heading = np.random.uniform(
                -self.max_rand_coordinate_heading, self.max_rand_coordinate_heading)
            past_motion_dropout = np.random.uniform(0, 1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        past_motion_dropout)
        transformed_img_feature = self.img_feature_transform(img_feature)

        offroad_mask = self.chauffeur_net_feature_generator.\
            render_offroad_mask(current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        ref_coords = [current_x,
                      current_y,
                      current_theta]
        pred_points = np.zeros((0, 4))
        pred_pose_dists = torch.rand(
            self.ouput_point_num, 1, self.img_size[1], self.img_size[0])
        pred_boxs = torch.rand(self.ouput_point_num, 1,
                               self.img_size[1], self.img_size[0])
        pred_obs = torch.rand(self.ouput_point_num, 1,
                              self.img_size[1], self.img_size[0])
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break

            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            pred_v = pred_point.trajectory_point.v
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

            gt_pose_dist = self.chauffeur_net_feature_generator.\
                render_gt_pose_dist(current_x,
                                    current_y,
                                    current_theta,
                                    frame.output.adc_future_trajectory_point,
                                    i,
                                    coordinate_heading)
            gt_pose_dist = self.img_bitmap_transform(gt_pose_dist)
            pred_pose_dists[i, :, :, :] = gt_pose_dist

            gt_pose_box = self.chauffeur_net_feature_generator.\
                render_gt_box(current_x,
                              current_y,
                              current_theta,
                              frame.output.adc_future_trajectory_point,
                              i,
                              coordinate_heading)
            gt_pose_box = self.img_bitmap_transform(gt_pose_box)
            pred_boxs[i, :, :, :] = gt_pose_box

            pred_obs_box = self.chauffeur_net_feature_generator.\
                render_obstacle_box_prediction_frame(current_x,
                                                     current_y,
                                                     current_theta,
                                                     frame.obstacle,
                                                     i,
                                                     coordinate_heading)
            pred_obs_box = self.img_bitmap_transform(pred_obs_box)
            pred_obs[i, :, :, :] = pred_obs_box

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        # draw agent current pose and box for hidden state intialization
        agent_current_box_img, agent_current_pose_img = self.chauffeur_net_feature_generator.\
            render_initial_agent_states(coordinate_heading)
        if self.img_bitmap_transform:
            agent_current_box_img = self.img_bitmap_transform(
                agent_current_box_img)
            agent_current_pose_img = self.img_bitmap_transform(
                agent_current_pose_img)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return ((transformed_img_feature,
                     agent_current_pose_img,
                     agent_current_box_img),
                    (pred_pose_dists,
                     pred_boxs,
                     torch.from_numpy(pred_points).float(),
                     pred_obs,
                     offroad_mask),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 agent_current_pose_img,
                 agent_current_box_img),
                (pred_pose_dists,
                 pred_boxs,
                 torch.from_numpy(pred_points).float(),
                 pred_obs,
                 offroad_mask))


if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.

    # dump one instance image for debug
    # dataset = TrajectoryImitationCNNDataset(
    #     '/apollo/data/2019-10-17-13-36-41/')
    config_file = "/fuel/fueling/planning/datasets/semantic_map_feature/" \
        "planning_semantic_map_config.pb.txt"
    imgs_dir = '/fuel/testdata/planning/semantic_map_features'
    dataset = TrajectoryImitationRNNDataset(
        '/apollo/data/output_data_evaluated/test/2019-10-17-13-36-41/complete',
        config_file,
        imgs_dir)

    dataset[50]
