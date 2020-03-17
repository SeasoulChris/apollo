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


class SemanticMapDataset(Dataset):
    def __init__(self, data_dir):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.instances = []

        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        # sort by filenames numerically: learning_data.<int>.bin.training_data.npy
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        region = None
        for file_path in all_file_paths:
            if 'training_data' not in file_path or 'bin' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            learning_data_frames = learning_data_pb2.LearningData
            with open(file_path, 'rb') as file_in:
                learning_data_frames.ParseFromString(file_in.read())
            for learning_data_frame in learning_data_frames.learning_data:
                self.instances.append(learning_data_frame)
            region = learning_data_frames.learning_data[0].map_name

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(
            region)

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
        for pred_point in frame.output.adc_future_traj_point:
            # TODO(Jinyun): evaluate whether use heading and acceleration
            local_coords = CoordUtils.world_to_relative(
                [pred_point.path_point.x, pred_point.path_point.y], ref_coords)
            heading_diff = pred_point.path_point.theta - ref_coords[2]
            pred_points.append(local_coords[1])
            pred_points.append(heading_diff)
            pred_points.append(pred_point.v)
            pred_points.append(pred_point.a)

        return (img, torch.from_numpy(np.asarray(pred_point.v).float())


if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.

    # dump one instance image for debug
    dataset=SemanticMapDataset('/fuel/fueling/planning/datasets/training')

    dataset[100]
