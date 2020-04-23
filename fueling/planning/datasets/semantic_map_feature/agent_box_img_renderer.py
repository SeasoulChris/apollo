#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils

class AgentBoxImgRenderer(object):
    """class of AgentBoxImgRenderer to create a image of ego car bounding box"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.front_edge_to_center = 3.89
        self.back_edge_to_center = 1.043
        self.left_edge_to_center = 1.055
        self.right_edge_to_center = 1.055
        self.east_oriented_box = np.array([[self.front_edge_to_center, self.front_edge_to_center,
                                            -self.back_edge_to_center, -self.back_edge_to_center],
                                           [self.left_edge_to_center, -self.right_edge_to_center,
                                            -self.right_edge_to_center, self.left_edge_to_center]]).T

    def draw_agent_box(self, coordinate_heading=0.):
        local_map = np.zeros(
            [self.local_size_h, self.local_size_w, 1], dtype=np.uint8)

        box_theta = np.pi / 2 + coordinate_heading
        theta = 0.
        corner_points = renderer_utils.box_affine_tranformation(self.east_oriented_box,
                                                                np.array([0, 0]),
                                                                box_theta,
                                                                np.array([0, 0]),
                                                                theta,
                                                                self.local_base_point_idx,
                                                                self.resolution)
        cv.fillPoly(local_map, [corner_points], color=(255))

        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    agentbox_renderer = AgentBoxImgRenderer(config_file)

    output_dir = './data_local_agent_box/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    cv.imwrite(os.path.join(output_dir, "agent_box.png"), agentbox_renderer.draw_agent_box(coordinate_heading=0.))
