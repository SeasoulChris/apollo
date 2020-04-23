#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils

class AgentPosesHistoryImgRenderer(object):
    """class of AgentPosesHistoryImgRenderer to create a image of past ego car poses"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y]) # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_history_time_horizon = config.max_ego_past_horizon  # second

    def draw_agent_poses_history(self, frame_time_sec, center_x,
                                 center_y, center_heading, ego_pose_history, coordinate_heading=0., past_motion_dropout=False):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        if past_motion_dropout:
            return local_map
            
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        current_time = frame_time_sec
        for i in range(len(ego_pose_history) - 1, -1, -1):
            ego_pose = ego_pose_history[i]
            relative_time = current_time - ego_pose.timestamp_sec
            if relative_time > self.max_history_time_horizon:
                break
            color = (1 - relative_time / self.max_history_time_horizon) * 255
            traj_point = tuple(renderer_utils.get_img_idx(
                renderer_utils.point_affine_transformation(
                    np.array([ego_pose.trajectory_point.path_point.x,
                              ego_pose.trajectory_point.path_point.y]),
                    self.local_base_point,
                    np.pi / 2 - self.local_base_heading + coordinate_heading),
                self.local_base_point_idx,
                self.resolution))
            cv.circle(local_map, tuple(traj_point),
                radius=4, color=color)
        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/test_interpolated_data/00001.record.0.bin.future_status.bin",
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
