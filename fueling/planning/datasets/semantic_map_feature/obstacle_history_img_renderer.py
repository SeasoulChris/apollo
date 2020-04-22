#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


class ObstacleHistoryImgRenderer(object):
    """class of ObstacleHistoryImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.max_history_length = config.max_obs_past_horizon  # second

    # TODO(Jinyun): evaluate whether use localization as current time
    def draw_obstacles(self, current_timestamp, obstacles):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        for obstacle in obstacles:
            if len(obstacle.obstacle_trajectory_point) == 0:
                continue
            current_time = obstacle.obstacle_trajectory_point[-1].timestamp_sec
            for i in range(len(obstacle.obstacle_trajectory_point) - 1, -1, -1):
                obstacle_history = obstacle.obstacle_trajectory_point[i]
                relative_time = current_time - obstacle_history.timestamp_sec
                if relative_time > self.max_history_length:
                    break
                points = np.zeros((0, 2))
                color = (1 - relative_time / self.max_history_length) * 255
                for point in obstacle_history.polygon_point:
                    point_idx = tuple(renderer_utils.get_img_idx(
                        renderer_utils.point_affine_transformation(
                            np.array([point.x, point.y]),
                            np.array([0, 0]),
                            np.pi / 2),
                        self.local_base_point_idx,
                        self.resolution))
                    points = np.vstack((points, point_idx))
                cv.fillPoly(local_map, [np.int32(points)], color=color)

        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/one_sample_test/learning_data.166.bin.future_status.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_obstacles/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    obstacle_mapping = ObstacleHistoryImgRenderer(config_file)
    for frame in offline_frames.learning_data:
        img = obstacle_mapping.draw_obstacles(
            frame.timestamp_sec, frame.obstacle)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
