#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv
import math

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class RoutingImgRenderer(object):
    """class of RoutingImgRenderer to create a image of routing of ego vehicle"""

    def __init__(self, config_file, region, map_path):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.center = None
        self.center_heading = None

        self.region = region
        self.hd_map = None
        self.map_path = map_path
        self.lane_dict = {}
        self._read_hdmap()
        self._load_lane()

    def _read_hdmap(self):
        """read the hdmap from base_map.bin"""
        self.hd_map = map_pb2.Map()
        map_path = self.map_path  # "/apollo/modules/map/data/" + self.region + "/base_map.bin"
        try:
            with open(map_path, 'rb') as file_in:
                self.hd_map.ParseFromString(file_in.read())
        except IOError:
            print("File at [" + map_path + "] is not accessible")
            exit()

    def _load_lane(self):
        for lane in self.hd_map.lane:
            self.lane_dict[lane.id.id] = lane

    def draw_local_routing(self, center_x, center_y, center_heading,
                           local_routing, coordinate_heading=0):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        if len(local_routing) == 0:
            print("No routing provided")
            return local_map
        routing_color_delta = int(255 / len(local_routing))
        for i in range(len(local_routing)):
            color = int(255 - i * routing_color_delta)
            if not self.lane_dict.__contains__(local_routing[i]):
                print("local routing lane is not found : " + local_routing[i])
                continue
            routing_lane = self.lane_dict[local_routing[i]]
            for segment in routing_lane.central_curve.segment:
                for i in range(len(segment.line_segment.point) - 1):
                    p0 = tuple(renderer_utils.get_img_idx(
                        renderer_utils.point_affine_transformation(
                            np.array([segment.line_segment.point[i].x,
                                      segment.line_segment.point[i].y]),
                            self.local_base_point,
                            np.pi / 2 - self.local_base_heading + coordinate_heading),
                        self.local_base_point_idx,
                        self.resolution))
                    p1 = tuple(renderer_utils.get_img_idx(
                        renderer_utils.point_affine_transformation(
                            np.array([segment.line_segment.point[i + 1].x,
                                      segment.line_segment.point[i + 1].y]),
                            self.local_base_point,
                            np.pi / 2 - self.local_base_heading + coordinate_heading),
                        self.local_base_point_idx,
                        self.resolution))
                    cv.line(local_map, tuple(p0), tuple(
                        p1), color=color, thickness=12)
        return local_map
