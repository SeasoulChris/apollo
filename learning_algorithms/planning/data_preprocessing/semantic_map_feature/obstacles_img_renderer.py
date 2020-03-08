#!/usr/bin/env python

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class ObstaclesImgRenderer(object):
    """class of ObstaclesImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self, current_timestamp):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) / 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.max_history_length = 8  # second
        self.current_timestamp = current_timestamp

    def _get_trans_point(self, p):
        # obstacles are in ego vehicle coordiantes where ego car faces toward EAST, so rotation to NORTH is done below
        theta = np.pi / 2
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_obstacles(self, obstacles):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        for obstacle in obstacles:
            for obstacle_history in obstacle.obstacle_trajectory_point:
                points = np.zeros((0, 2))
                color = (255 - min((self.current_timestamp - obstacle_history.timestamp_sec),
                                  self.max_history_length) / self.max_history_length * 255)
                for point in obstacle_history.polygon_point:
                    point = self._get_trans_point(
                        [point.x, point.y])
                    points = np.vstack((points, point))
                cv.fillPoly(local_map, [np.int32(points)], color=color)

        return local_map
