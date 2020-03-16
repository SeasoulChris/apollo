#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class ObstaclesImgRenderer(object):
    """class of ObstaclesImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) / 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.max_history_length = 8  # second
        self.current_timestamp = None

    def _get_trans_point(self, p):
        # obstacles are in ego vehicle coordiantes where ego car faces toward EAST, so rotation to NORTH is done below
        theta = np.pi / 2
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_obstacles(self, current_timestamp, obstacles):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.current_timestamp = current_timestamp
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


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_obstacles/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    obstacle_mapping = ObstaclesImgRenderer()
    for frame in offline_frames.learning_data:
        img = obstacle_mapping.draw_obstacles(
            frame.timestamp_sec, frame.obstacle)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
