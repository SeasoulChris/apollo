#!/usr/bin/env python
import os

import numpy as np
import cv2 as cv
import pyproj

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


class BaseSpeedLimitImgRenderer(object):
    """
    class of BaseSpeedLimitImgRenderer to get a feature map according to Baidu Apollo Map Format
    """

    def __init__(self, config_file, region):
        """contruct function to init BaseRoadMapImgRenderer object"""
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution   # in meter/pixel
        self.base_map_padding = config.base_map_padding    # in meter

        self.region = region

        self.city_driving_max_speed = config.city_driving_max_speed  # 22.22 m/s /approx 80km/h

        self.base_point = None
        self.GRID = None
        self.base_map = None

        self._read_hdmap()
        self._build_canvas()
        self._draw_base_map()
        print("Base Speed Limit Map base point is "
              + str(self.base_point[0]) + ", " + str(self.base_point[1]))
        print("Base Speed Limit Map W * H is "
              + str(self.GRID[0]) + " * " + str(self.GRID[1]))

    def _read_hdmap(self):
        """read the hdmap from base_map.bin"""
        self.hd_map = map_pb2.Map()
        map_path = "/apollo/modules/map/data/" + self.region + "/base_map.bin"
        try:
            with open(map_path, 'rb') as file_in:
                self.hd_map.ParseFromString(file_in.read())
        except IOError:
            print("File at [" + map_path + "] is not accessible")
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
            [self.GRID[1], self.GRID[0], 3], dtype=np.uint8)

    def _draw_base_map(self):
        self._draw_speed_limit()
        self._draw_speed_bump()

    def get_speedlimit_coloring(self, speed_limit):
        green_level = (
            speed_limit / self.city_driving_max_speed) * (255 - 64) + 64
        color = (0, green_level, 0)
        return color

    def _draw_speed_limit(self):
        for lane in self.hd_map.lane:
            speedlimit = lane.speed_limit
            for segment in lane.central_curve.segment:
                for i in range(len(segment.line_segment.point) - 1):
                    p0 = renderer_utils.get_img_idx(np.array(
                        [segment.line_segment.point[i].x,
                         segment.line_segment.point[i].y]) - self.base_point,
                        self.base_point_idx,
                        self.resolution)
                    p1 = renderer_utils.get_img_idx(np.array(
                        [segment.line_segment.point[i + 1].x,
                         segment.line_segment.point[i + 1].y]) - self.base_point,
                        self.base_point_idx,
                        self.resolution)
                    cv.line(self.base_map, tuple(p0), tuple(p1),
                            color=self.get_speedlimit_coloring(speedlimit), thickness=4)

    def _draw_speed_bump(self, color=(255, 0, 0)):
        for speed_bump in self.hd_map.speed_bump:
            for position in speed_bump.position:
                for segment in position.segment:
                    for i in range(len(segment.line_segment.point) - 1):
                        p0 = renderer_utils.get_img_idx(np.array(
                            [segment.line_segment.point[i].x,
                             segment.line_segment.point[i].y]) - self.base_point,
                            self.base_point_idx,
                            self.resolution)
                        p1 = renderer_utils.get_img_idx(np.array(
                            [segment.line_segment.point[i + 1].x,
                             segment.line_segment.point[i + 1].y]) - self.base_point,
                            self.base_point_idx,
                            self.resolution)
                        cv.line(self.base_map, tuple(p0), tuple(
                            p1), color=color, thickness=12)


if __name__ == '__main__':
    imgs_dir = "/fuel/testdata/planning/semantic_map_features"
    config_file = "/fuel/fueling/planning/datasets/semantic_map_feature/" \
        "planning_semantic_map_config.pb.txt"
    mapping = BaseSpeedLimitImgRenderer(
        config_file, "sunnyvale_with_two_offices")
    # using cv.imwrite to .png so we can simply use cv.imread and get the exactly same matrix
    cv.imwrite(os.path.join(imgs_dir, mapping.region
                            + "_speedlimit.png"), mapping.base_map)
