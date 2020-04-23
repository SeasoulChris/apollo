#!/usr/bin/env python

import cv2 as cv
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from modules.planning.proto import planning_semantic_map_config_pb2
from modules.planning.proto import learning_data_pb2

from fueling.common.coord_utils import CoordUtils
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
from fueling.planning.datasets.semantic_map_feature.chauffeur_net_feature_generator import ChauffeurNetFeatureGenerator


class TrajectoryImitationCNNDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir,
                 input_data_agumentation=False, ouput_point_num=10):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.instances = []

        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        # sort by filenames numerically: learning_data.<int>.bin.training_data.bin
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        region = None
        for file_path in all_file_paths:
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            learning_data_frames = learning_data_pb2.LearningData()
            with open(file_path, 'rb') as file_in:
                learning_data_frames.ParseFromString(file_in.read())
            for learning_data_frame in learning_data_frames.learning_data:
                self.instances.append(learning_data_frame)
            region = learning_data_frames.learning_data[0].map_name

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region)
        self.input_data_agumentation = input_data_agumentation
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.ouput_point_num = ouput_point_num

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame = self.instances[idx]

        coordinate_heading = 0.
        past_motion_dropout = False
        if self.input_data_agumentation:
            coordinate_heading = np.random.uniform(
                -self.max_rand_coordinate_heading, self.max_rand_coordinate_heading)
            past_motion_dropout = np.random.uniform(0, 1) > 0.5

        img = self.chauffeur_net_feature_generator.render_stacked_img_features(frame.frame_num,
                                                                               frame.adc_trajectory_point[-1].timestamp_sec,
                                                                               frame.adc_trajectory_point,
                                                                               frame.obstacle,
                                                                               frame.localization.position.x,
                                                                               frame.localization.position.y,
                                                                               frame.localization.heading,
                                                                               frame.routing.local_routing_lane_id,
                                                                               frame.traffic_light_detection.traffic_light,
                                                                               coordinate_heading,
                                                                               past_motion_dropout)

        if self.img_transform:
            img = self.img_transform(img)

        ref_coords = [frame.localization.position.x,
                      frame.localization.position.y,
                      frame.localization.heading]
        pred_points = []
        for pred_point in frame.output.adc_future_trajectory_point:
            # TODO(Jinyun): validate future trajectory points size and deltaT, ouput_point_num
            # points, 5 attributes
            if len(pred_points) >= self.ouput_point_num * 5:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            pred_v = pred_point.trajectory_point.v
            pred_a = pred_point.trajectory_point.a
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_points.append(local_coords[0])
            pred_points.append(local_coords[1])
            pred_points.append(heading_diff)
            pred_points.append(pred_v)
            pred_points.append(pred_a)

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if len(pred_points) < self.ouput_point_num * 5:
            return self.__getitem__(idx - 1)

        return (img, torch.from_numpy(np.asarray(pred_points)).float())


class TrajectoryImitationRNNDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir,
                 input_data_agumentation=False, ouput_point_num=10):
        # TODO(Jinyun): refine transform function
        self.img_feature_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.img_bitmap_transform = transforms.Compose([
            # Normalized to [0, 1]
            transforms.ToTensor()])
        self.instances = []

        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        # sort by filenames numerically: learning_data.<int>.bin.training_data.bin
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        region = None
        for file_path in all_file_paths:
            print(file_path)
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            learning_data_frames = learning_data_pb2.LearningData()
            with open(file_path, 'rb') as file_in:
                learning_data_frames.ParseFromString(file_in.read())
            for learning_data_frame in learning_data_frames.learning_data:
                self.instances.append(learning_data_frame)
            region = learning_data_frames.learning_data[0].map_name

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region,
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

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame = self.instances[idx]

        coordinate_heading = 0.
        past_motion_dropout = False
        if self.input_data_agumentation:
            coordinate_heading = np.random.uniform(
                -self.max_rand_coordinate_heading, self.max_rand_coordinate_heading)
            past_motion_dropout = np.random.uniform(0, 1) > 0.5

        img_feature = self.chauffeur_net_feature_generator.render_stacked_img_features(frame.frame_num,
                                                                                       frame.adc_trajectory_point[-1].timestamp_sec,
                                                                                       frame.adc_trajectory_point,
                                                                                       frame.obstacle,
                                                                                       frame.localization.position.x,
                                                                                       frame.localization.position.y,
                                                                                       frame.localization.heading,
                                                                                       frame.routing.local_routing_lane_id,
                                                                                       frame.traffic_light_detection.traffic_light,
                                                                                       coordinate_heading,
                                                                                       past_motion_dropout)
        if self.img_feature_transform:
            img_feature = self.img_feature_transform(img_feature)

        offroad_mask = self.chauffeur_net_feature_generator.render_offroad_mask(frame.localization.position.x,
                                                                                frame.localization.position.y,
                                                                                frame.localization.heading,
                                                                                coordinate_heading)
        if self.img_bitmap_transform:
            offroad_mask = self.img_bitmap_transform(offroad_mask)

        ref_coords = [frame.localization.position.x,
                      frame.localization.position.y,
                      frame.localization.heading]
        pred_points = np.zeros((0, 4))
        pred_pose_dists = torch.rand(self.ouput_point_num, 1, self.img_size[1], self.img_size[0])
        pred_boxs = torch.rand(self.ouput_point_num, 1, self.img_size[1], self.img_size[0])
        pred_obs = torch.rand(self.ouput_point_num, 1, self.img_size[1], self.img_size[0])
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            # TODO(Jinyun): validate future trajectory points size and deltaT,
            # ouput_point_num points
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

            gt_pose_dist = self.chauffeur_net_feature_generator.render_gt_pose_dist(frame.localization.position.x,
                                                                                    frame.localization.position.y,
                                                                                    frame.localization.heading,
                                                                                    frame.output.adc_future_trajectory_point,
                                                                                    i,
                                                                                    coordinate_heading)
            if self.img_bitmap_transform:
                gt_pose_dist = self.img_bitmap_transform(gt_pose_dist)
            pred_pose_dists[i, :, :, :] = gt_pose_dist

            gt_pose_box = self.chauffeur_net_feature_generator.render_gt_box(frame.localization.position.x,
                                                                             frame.localization.position.y,
                                                                             frame.localization.heading,
                                                                             frame.output.adc_future_trajectory_point,
                                                                             i,
                                                                             coordinate_heading)
            if self.img_bitmap_transform:
                gt_pose_box = self.img_bitmap_transform(gt_pose_box)
            pred_boxs[i, :, :, :] = gt_pose_box

            pred_obs_box = self.chauffeur_net_feature_generator.render_obstacle_box_prediction_frame(frame.localization.position.x,
                                                                                                     frame.localization.position.y,
                                                                                                     frame.localization.heading,
                                                                                                     frame.obstacle,
                                                                                                     i,
                                                                                                     coordinate_heading)
            if self.img_bitmap_transform:
                pred_obs_box = self.img_bitmap_transform(pred_obs_box)
            pred_obs[i, :, :, :] = pred_obs_box

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        return (img_feature,
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
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    imgs_dir = '/fuel/testdata/planning/semantic_map_features'
    dataset = TrajectoryImitationRNNDataset(
        '/apollo/data/output_data_evaluated/test/2019-10-17-13-36-41/complete', config_file, imgs_dir)

    dataset[50]
