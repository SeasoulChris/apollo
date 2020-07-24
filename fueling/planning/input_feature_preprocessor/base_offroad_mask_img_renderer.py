#!/usr/bin/env python
import os

import numpy as np
import cv2 as cv
import pyproj

from modules.map.proto import map_pb2
from modules.map.proto import map_road_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class BaseOffroadMaskImgRenderer(object):
    """class of BaseRoadMapImgRenderer to get a feature map according to Baidu Apollo Map Format"""

    def __init__(self, config_file, region, base_map_data_dir):
        """contruct function to init BaseRoadMapImgRenderer object"""
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution   # in meter/pixel
        self.base_map_padding = config.base_map_padding    # in meter

        self.region = region
        self.map_path = os.path.join(os.path.join(
            base_map_data_dir, region), "base_map.bin")

        self.base_point = None
        self.GRID = None
        self.base_map = None

        self._read_hdmap()
        self._build_canvas()
        self._draw_base_map()
        logging.info("Base Offroad Mask Map base point is "
                     + str(self.base_point[0]) + ", " + str(self.base_point[1]))
        logging.info("Base Offroad Mask Map W * H is "
                     + str(self.GRID[0]) + " * " + str(self.GRID[1]))

    def _read_hdmap(self):
        """read the hdmap from base_map.bin"""
        self.hd_map = map_pb2.Map()
        try:
            with open(self.map_path, 'rb') as file_in:
                self.hd_map.ParseFromString(file_in.read())
        except IOError:
            logging.error("File at [" + self.map_path + "] is not accessible")
            exit()

    def _build_canvas(self):
        """build canvas in np.array with padded left bottom point as base_point"""
        projection_rule = self.hd_map.header.projection.proj
        projector = pyproj.Proj(projection_rule, preserve_units=True)
        left_bottom_x, left_bottom_y = projector(self.hd_map.header.left,
                                                 self.hd_map.header.bottom)
        right_top_x, right_top_y = projector(self.hd_map.header.right,
                                             self.hd_map.header.top)

        # TODO(Jinyun): resolve opencv can't open lager than 1 Gigapixel issue
        # (Map "sunnyvle" is too large)
        if self.region == "sunnyvale":
            left_bottom_x = 585975.3316302994
            left_bottom_y = 4140016.6342316796
            right_top_x = 588538.5457265645
            right_top_y = 4141747.6943244375

        self.base_point = np.array([left_bottom_x - self.base_map_padding,
                                    left_bottom_y - self.base_map_padding])
        self.GRID = [int(np.round((right_top_x - left_bottom_x
                                   + 2 * self.base_map_padding) / self.resolution)),
                     int(np.round((right_top_y - left_bottom_y
                                   + 2 * self.base_map_padding) / self.resolution))]
        self.base_point_idx = np.array([0, self.GRID[1]])
        self.base_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)

    def _draw_base_map(self):
        self._draw_road()

    def _draw_road(self, color=(0)):
        self.base_map = (self.base_map + 1) * 255
        for road in self.hd_map.road:
            for section in road.section:
                points = np.zeros((0, 2))
                for edge in section.boundary.outer_polygon.edge:
                    if edge.type == map_road_pb2.BoundaryEdge.Type.LEFT_BOUNDARY:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point)):
                                point = renderer_utils.get_img_idx(np.array(
                                    [segment.line_segment.point[i].x,
                                     segment.line_segment.point[i].y]) - self.base_point,
                                    self.base_point_idx,
                                    self.resolution)
                                points = np.vstack((points, point))
                    elif edge.type == map_road_pb2.BoundaryEdge.Type.RIGHT_BOUNDARY:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point) - 1, -1, -1):
                                point = renderer_utils.get_img_idx(np.array(
                                    [segment.line_segment.point[i].x,
                                     segment.line_segment.point[i].y]) - self.base_point,
                                    self.base_point_idx,
                                    self.resolution)
                                points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)
