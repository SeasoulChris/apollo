#!/usr/bin/env python
import numpy as np
import cv2 as cv

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class AgentPosesHistoryImgRenderer(object):
    """class of AgentPosesHistoryImgRenderer to create a image of past ego car poses"""

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
        self.max_history_time_horizon = config.max_ego_past_horizon  # second

        self.current_pose_img = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.current_pose_img[self.local_base_point_idx[1], self.local_base_point_idx[0]] = 255

    def draw_agent_poses_history(self, current_timestamp, center_x,
                                 center_y, center_heading, ego_pose_history,
                                 coordinate_heading=0., past_motion_dropout=False):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        if past_motion_dropout:
            return local_map

        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        for i in range(len(ego_pose_history)):
            ego_pose = ego_pose_history[i]
            relative_time = current_timestamp - ego_pose.timestamp_sec
            if relative_time > self.max_history_time_horizon:
                continue
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
                      radius=2, color=color, thickness=-1)
        return local_map

    def draw_agent_current_pose(self):
        return self.current_pose_img
