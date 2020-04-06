#!/usr/bin/env python

import cv2 as cv
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import fueling.common.logging as logging
from fueling.common.coord_utils import CoordUtils
import fueling.common.file_utils as file_utils
from fueling.planning.datasets.semantic_map_feature.chauffeur_net_feature_generator import ChauffeurNetFeatureGenerator
from modules.planning.proto import learning_data_pb2


class TrajectoryImitationCNNDataset(Dataset):
    def __init__(self, data_dir):
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
        base_map_update_flag = False
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(
            region, base_map_update_flag)

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame = self.instances[idx]

        img = self.chauffeur_net_feature_generator.render_stacked_img_features(frame.frame_num,
                                                                               frame.timestamp_sec,
                                                                               frame.adc_trajectory_point,
                                                                               frame.obstacle,
                                                                               frame.localization.position.x,
                                                                               frame.localization.position.y,
                                                                               frame.localization.heading,
                                                                               frame.routing.local_routing_lane_id,
                                                                               frame.traffic_light)

        if self.img_transform:
            img = self.img_transform(img)

        ref_coords = [frame.localization.position.x,
                      frame.localization.position.y,
                      frame.localization.heading]
        pred_points = []
        for pred_point in frame.output.adc_future_trajectory_point:
            # TODO(Jinyun): validate future trajectory points size and deltaT, 30 points, 5 attributes
            if len(pred_points) >= 30 * 5:
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
        if len(pred_points) < 30 * 5:
            return self.__getitem__(idx - 1)

        return (img, torch.from_numpy(np.asarray(pred_points)).float())


class TrajectoryImitationRNNDataset(Dataset):
    def __init__(self, data_dir):
        # TODO(Jinyun): refine transform function
        self.img_feature_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.img_pred_transform = transforms.Compose([
            # Normalized to [0, 1] is good enough
            transforms.ToTensor()])
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
        base_map_update_flag = False
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(
            region, base_map_update_flag)

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame = self.instances[idx]

        img_feature = self.chauffeur_net_feature_generator.render_stacked_img_features(frame.frame_num,
                                                                                       frame.timestamp_sec,
                                                                                       frame.adc_trajectory_point,
                                                                                       frame.obstacle,
                                                                                       frame.localization.position.x,
                                                                                       frame.localization.position.y,
                                                                                       frame.localization.heading,
                                                                                       frame.routing.local_routing_lane_id,
                                                                                       frame.traffic_light)
        if self.img_feature_transform:
            img_feature = self.img_feature_transform(img_feature)

        ref_coords = [frame.localization.position.x,
                      frame.localization.position.y,
                      frame.localization.heading]
        pred_points = np.zeros((0, 4))
        pred_pose_dists = torch.rand(30, 1 , 224, 224)
        pred_boxs = torch.rand(30, 1, 224, 224)
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            # TODO(Jinyun): validate future trajectory points size and deltaT, 30 points
            if i + 1 > 30:
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
                                                                                    i)
            if self.img_pred_transform:
                gt_pose_dist = self.img_pred_transform(gt_pose_dist)
            pred_pose_dists[i,:,:,:] = gt_pose_dist

            gt_pose_box = self.chauffeur_net_feature_generator.render_gt_box(frame.localization.position.x,
                                                                             frame.localization.position.y,
                                                                             frame.localization.heading,
                                                                             frame.output.adc_future_trajectory_point,
                                                                             i)
            if self.img_pred_transform:
                gt_pose_box = self.img_pred_transform(gt_pose_box)
            pred_boxs[i,:,:,:] = gt_pose_box

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < 30:
            return self.__getitem__(idx - 1)

        return (img_feature,
                (pred_pose_dists,
                pred_boxs,
                torch.from_numpy(pred_points).float()))


if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.

    # dump one instance image for debug
    # dataset = TrajectoryImitationCNNDataset(
    #     '/apollo/data/2019-10-17-13-36-41/')

    dataset = TrajectoryImitationRNNDataset(
        '/apollo/data/2019-10-17-13-36-41/')

    dataset[50]
