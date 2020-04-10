#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class ObstaclePredictionsImgRenderer(object):
    """class of ObstaclesImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) // 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_prediction_time_horizon = 3  # second

    def _get_affine_points(self, p):
        # obstacles are in ego vehicle coordiantes where ego car faces toward EAST, so rotation to NORTH is done below
        theta = np.pi / 2
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_obstacle_prediction(self, center_x, center_y, center_heading, obstacles):
        # TODO(Jinyun): make use of multi-modal and probability
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [center_x, center_y])
        self.local_base_heading = center_heading

        for obstacle in obstacles:
            if obstacle.HasField("obstacle_prediction") and len(obstacle.obstacle_prediction.trajectory) > 0:
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

    def draw_obstacle_prediction_frame(self, center_x, center_y, center_heading, obstacles, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [center_x, center_y])
        self.local_base_heading = center_heading

        for obstacle in obstacles:
            if obstacle.HasField("obstacle_prediction") and len(obstacle.obstacle_prediction.trajectory) > 0:
                max_prob_idx = 0
                max_prob = 0
                for i in range(len(obstacle.obstacle_prediction.trajectory)):
                    trajectory = obstacle.obstacle_prediction.trajectory[i]
                    if trajectory.probability > max_prob:
                        max_prob_idx = i
                        max_prob = trajectory.probability
                if len(obstacle.obstacle_prediction.trajectory[max_prob_idx].trajectory_point) <= timestamp_idx:
                    print("timestamp_idx larger than what is available in obstacle prediction")
                else:
                    trajectory_point = obstacle.obstacle_prediction.trajectory[max_prob_idx].trajectory_point[timestamp_idx]
                    cv.circle(local_map, tuple(self._get_affine_points(
                        np.array([trajectory_point.path_point.x, trajectory_point.path_point.y]))), radius=4, color=255)

        return local_map

if __name__ == "__main__":
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
    obstacle_predictions_mapping = ObstaclePredictionsImgRenderer()
    for frame in offline_frames.learning_data:
        img = obstacle_predictions_mapping.draw_obstacle_prediction(
            frame.localization.position.x,
            frame.localization.position.y, frame.localization.heading, frame.obstacle)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
