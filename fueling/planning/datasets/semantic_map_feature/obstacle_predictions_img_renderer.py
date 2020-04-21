#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils

class ObstaclePredictionsImgRenderer(object):
    """class of ObstaclesImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = config.ego_idx_x
        self.local_base_point_h_idx = config.ego_idx_y  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_prediction_time_horizon = config.max_obs_future_horizon  # second

    def _get_trans_point(self, p):
        p = np.round(p / self.resolution)
        return [self.local_base_point_w_idx + int(p[0]), self.local_base_point_h_idx - int(p[1])]

    def _get_affine_points(self, p):
        # obstacles are in ego vehicle coordiantes where ego car faces toward
        # EAST, so rotation to NORTH is done below
        theta = np.pi / 2
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx +
                int(point[0]), self.local_base_point_h_idx - int(point[1])]

    # TODO(Jinyun): move to utils
    def _get_affine_prediction_box(self, p, box_theta, box_length, box_width):
        p = p - self.local_base_point
        theta = np.pi / 2 - self.local_base_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        box_theta_diff = np.pi / 2 + box_theta - self.local_base_heading
        corner_points = np.dot(np.array([[np.cos(box_theta_diff), -np.sin(box_theta_diff)],
                                         [np.sin(box_theta_diff), np.cos(box_theta_diff)]]),
                               np.array([[box_length / 2, box_length / 2,
                                          -box_length / 2, box_length / 2],
                                         [box_width, -box_width,
                                          -box_width, box_width]])).T + point
        corner_points = [self._get_trans_point(
            point) for point in corner_points]
        return np.int32(corner_points)

    def draw_obstacle_prediction(self, center_x, center_y, center_heading, obstacles):
        # TODO(Jinyun): make use of multi-modal and probability
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [center_x, center_y])
        self.local_base_heading = center_heading

        for obstacle in obstacles:
            if obstacle.HasField("obstacle_prediction") and len(
                    obstacle.obstacle_prediction.trajectory) > 0:
                max_prob_idx = 0
                max_prob = 0
                for i in range(len(obstacle.obstacle_prediction.trajectory)):
                    trajectory = obstacle.obstacle_prediction.trajectory[i]
                    if trajectory.probability > max_prob:
                        max_prob_idx = i
                        max_prob = trajectory.probability
                for trajectory_point in obstacle.obstacle_prediction.trajectory[max_prob_idx].trajectory_point:
                    if trajectory_point.relative_time > self.max_prediction_time_horizon:
                        break
                    color = trajectory_point.relative_time / self.max_prediction_time_horizon * 255
                    cv.circle(local_map, tuple(self._get_affine_points(
                        np.array([trajectory_point.path_point.x, trajectory_point.path_point.y]))), radius=4, color=color)

        return local_map

    def draw_obstacle_box_prediction_frame(
            self, center_x, center_y, center_heading, obstacles, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [center_x, center_y])
        self.local_base_heading = center_heading

        for obstacle in obstacles:
            obs_length = obstacle.length
            obs_width = obstacle.width
            if obstacle.HasField("obstacle_prediction") and len(
                    obstacle.obstacle_prediction.trajectory) > 0:
                max_prob_idx = 0
                max_prob = 0
                for i in range(len(obstacle.obstacle_prediction.trajectory)):
                    trajectory = obstacle.obstacle_prediction.trajectory[i]
                    if trajectory.probability > max_prob:
                        max_prob_idx = i
                        max_prob = trajectory.probability
                if len(
                        obstacle.obstacle_prediction.trajectory[max_prob_idx].trajectory_point) <= timestamp_idx:
                    print("timestamp_idx larger than what is available in obstacle prediction")
                else:
                    path_point = obstacle.obstacle_prediction.trajectory[
                        max_prob_idx].trajectory_point[timestamp_idx].path_point
                    corner_points = self._get_affine_prediction_box(
                        np.array([path_point.x, path_point.y]), path_point.theta, obs_length, obs_width)
                    for corner_point in corner_points:
                        if corner_point[0] < 0 or corner_point[0] > self.local_size_h or corner_point[1] < 0 or corner_point[1] > self.local_size_h:
                            print("draw_agent_box_future out of canvas bound")
                            return local_map
                    cv.fillPoly(local_map, [corner_points], color=255)

        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_obstacle_predictions/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    obstacle_predictions_mapping = ObstaclePredictionsImgRenderer(config_file)
    for frame in offline_frames.learning_data:
        img = obstacle_predictions_mapping.draw_obstacle_prediction(
            frame.localization.position.x,
            frame.localization.position.y, frame.localization.heading, frame.obstacle)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
