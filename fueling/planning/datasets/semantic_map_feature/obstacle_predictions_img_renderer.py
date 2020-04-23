#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


class ObstaclePredictionsImgRenderer(object):
    """class of ObstaclesImgRenderer to create images of surrounding obstacles with bounding boxes"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_prediction_time_horizon = config.max_obs_future_horizon  # second

    def draw_obstacle_prediction(self, center_x, center_y, center_heading, obstacles, coordinate_heading=0.):
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
                    # obstacles are in ego vehicle coordiantes where ego car faces toward
                    # EAST, so rotation to NORTH is done below
                    trajectory_idx = tuple(renderer_utils.get_img_idx(
                        renderer_utils.point_affine_transformation(
                            np.array([trajectory_point.path_point.x,
                                      trajectory_point.path_point.y]),
                            np.array([0, 0]),
                            np.pi / 2 + coordinate_heading),
                        self.local_base_point_idx,
                        self.resolution))
                    cv.circle(local_map, tuple(trajectory_idx),
                              radius=4, color=color)

        return local_map

    def draw_obstacle_box_prediction_frame(
            self, center_x, center_y, center_heading, obstacles, timestamp_idx, coordinate_heading=0.):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [center_x, center_y])
        self.local_base_heading = center_heading

        for obstacle in obstacles:
            box_length = obstacle.length
            box_width = obstacle.width
            if obstacle.HasField("obstacle_prediction") and len(
                    obstacle.obstacle_prediction.trajectory) > 0:
                max_prob_idx = 0
                max_prob = 0
                for i in range(len(obstacle.obstacle_prediction.trajectory)):
                    trajectory = obstacle.obstacle_prediction.trajectory[i]
                    if trajectory.probability > max_prob:
                        max_prob_idx = i
                        max_prob = trajectory.probability
                if len(obstacle.obstacle_prediction.trajectory[max_prob_idx].trajectory_point) <= timestamp_idx:
                    # print("timestamp_idx larger than what is available in obstacle prediction")
                    continue
                else:
                    path_point = obstacle.obstacle_prediction.trajectory[
                        max_prob_idx].trajectory_point[timestamp_idx].path_point
                    path_point_array = np.array([path_point.x, path_point.y])
                    east_oriented_box = np.array([[box_length / 2, box_length / 2,
                                                   -box_length / 2, box_length / 2],
                                                  [box_width, -box_width,
                                                   -box_width, box_width]]).T

                    # obstacles are in ego vehicle coordiantes where ego car faces toward
                    # EAST, so rotation to NORTH is done below
                    corner_points = renderer_utils.box_affine_tranformation(east_oriented_box,
                                                                            path_point_array,
                                                                            np.pi / 2 + path_point.theta + coordinate_heading,
                                                                            np.array([0, 0]),
                                                                            np.pi / 2 + coordinate_heading,
                                                                            self.local_base_point_idx,
                                                                            self.resolution)

                    Prediction_out_of_bound = False
                    for corner_point in corner_points:
                        if corner_point[0] < 0 or corner_point[0] > self.local_size_h or corner_point[1] < 0 or corner_point[1] > self.local_size_h:
                            # print("draw_agent_box_future out of canvas bound")
                            Prediction_out_of_bound = True
                            break
                    if not Prediction_out_of_bound:
                        cv.fillPoly(local_map, [corner_points], color=255)

        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/test_interpolated_data/00001.record.0.bin.future_status.bin", 'rb') as file_in:
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
