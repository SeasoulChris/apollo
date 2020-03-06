#!/usr/bin/env python

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class AgentPosesHistoryImgRenderer(object):
    """class of AgentPosesHistoryImgRenderer to create a image of past ego car poses"""

    def __init__(self):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) / 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None

    def _get_trans_point(self, p):
        point = np.round((p - self.local_base_point) / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_agent_poses_history(self, ego_pose_history):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [ego_pose_history[0][0], ego_pose_history[0][1]])
        for ego_pose in ego_pose_history:
            cv.circle(local_map, self._get_trans_point(
                np.array(ego_pose)), color=(255), thickness=4)
        return local_map
