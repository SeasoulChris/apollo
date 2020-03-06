#!/usr/bin/env python

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class AgentBoxImgRenderer(object):
    """class of AgentBoxImgRenderer to create a image of ego car bounding box"""

    def __init__(self):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) / 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        # TODO(Jinyun): read vehicle param from elsewhere
        self.front_edge_to_center = 3.89
        self.back_edge_to_center = 1.043
        self.left_edge_to_center = 1.055
        self.right_edge_to_center = 1.055

        self.right_width = int(self.right_edge_to_center / self.resolution)
        self.left_width = int(self.left_edge_to_center / self.resolution)
        self.front_length = int(self.front_edge_to_center / self.resolution)
        self.back_length = int(self.back_edge_to_center / self.resolution)

        self.local_map = np.zeros(
            [self.local_size_h, self.local_size_w, 1], dtype=np.uint8)

        front_right_connor = [self.local_base_point_h_idx -
                              self.front_length, self.local_base_point_w_idx + self.right_width]
        front_left_connor = [self.local_base_point_h_idx -
                             self.front_length, self.local_base_point_w_idx - self.right_width]
        back_left_connor = [self.local_base_point_h_idx +
                            self.back_length, self.local_base_point_w_idx - self.right_width]
        back_right_connor = [self.local_base_point_h_idx +
                             self.back_length, self.local_base_point_w_idx + self.right_width]

        cv.fillPoly(self.local_map, [np.int32(
            [front_right_connor, front_left_connor, back_left_connor, back_right_connor])], color=(255))

    def draw_agent_box(self):
        return self.local_map
