#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils


class AgentPosesHistoryImgRenderer(object):
    """class of AgentPosesHistoryImgRenderer to create a image of past ego car poses"""

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
        self.max_history_time_horizon = config.max_ego_past_horizon  # second

    def _get_affine_points(self, p):
        p = p - self.local_base_point
        theta = np.pi / 2 - self.local_base_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx +
                int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_agent_poses_history(self, frame_time_sec, center_x,
                                 center_y, center_heading, ego_pose_history):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        current_time = frame_time_sec
        for i in range(len(ego_pose_history) - 1, -1, -1):
            ego_pose = ego_pose_history[i]
            relative_time = current_time - ego_pose.timestamp_sec
            if relative_time > self.max_history_time_horizon:
                break
            color = (1 - relative_time / self.max_history_time_horizon) * 255
            cv.circle(local_map, tuple(self._get_affine_points(
                np.array([ego_pose.trajectory_point.path_point.x, ego_pose.trajectory_point.path_point.y]))),
                radius=4, color=color)
        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/2019-10-17-13-36-41/ver0/valid_set/learning_data.66.bin.future_status.bin",
              'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_agent_pose_history/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    agent_history_mapping = AgentPosesHistoryImgRenderer(config_file)
    for frame in offline_frames.learning_data:
        img = agent_history_mapping.draw_agent_poses_history(frame.timestamp_sec,
                                                             frame.localization.position.x,
                                                             frame.localization.position.y,
                                                             frame.localization.heading,
                                                             frame.adc_trajectory_point)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
