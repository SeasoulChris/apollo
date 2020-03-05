#!/usr/bin/env python

import numpy as np
import cv2 as cv
import pyproj

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2

class BaseSpeedLimitImgRenderer(object):
    """class of BaseSpeedLimitImgRenderer to get a feature map according to Baidu Apollo Map Format"""

    def __init__(self, region):
        """contruct function to init BaseSpeedLimitImgRenderer object"""
        self.region = region
        # TODO(Jinyun): use config file
        self.resolution = 0.1   # in meter/pixel
        self.base_map_padding = 100    # in meter
        self.city_driving_max_speed = 22.22 # 22.22 m/s /approx 80km/h

        self.base_point = None
        self.GRID = None
        self.base_map = None

        self._read_hdmap()
        self._build_canvas()
        self._draw_base_map()

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
        self.base_point = np.array([left_bottom_x - self.base_map_padding,
                                    left_bottom_y - self.base_map_padding])
        self.GRID = [int(np.round((right_top_x - left_bottom_x + 2 * self.base_map_padding) / self.resolution)),
                     int(np.round((right_top_y - left_bottom_y + 2 * self.base_map_padding) / self.resolution))]
        self.base_map = np.zeros(
            [self.GRID[1], self.GRID[0], 3], dtype=np.uint8)

    def _draw_base_map(self):
        self._draw_speed_limit()
        self._draw_speed_bump()

    def get_trans_point(self, p):
        point = np.round((p - self.base_point) / self.resolution)
        return [int(point[0]), self.GRID[1] - int(point[1])]

    def get_speedlimit_coloring(self, speed_limit):
        green_level = (speed_limit / self.city_driving_max_speed) * (255 - 64) + 64
        color = (0, green_level, 0)
        return color

    def _draw_speed_limit(self):
        for lane in self.hd_map.lane:
            if lane.left_boundary.virtual and lane.right_boundary.virtual:
                continue
            speedlimit = lane.speed_limit
            for segment in lane.central_curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point(
                        [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point(
                        [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    cv.line(self.base_map, tuple(p0), tuple(p1),
                            color=self.get_speedlimit_coloring(speedlimit), thickness=4)

    def _draw_speed_bump(self, color=(255, 0, 0)):
        for speed_bump in self.hd_map.speed_bump:
            for position in speed_bump.position:
                for segment in position.segment:
                    for i in range(len(segment.line_segment.point)-1):
                        p0 = self.get_trans_point(
                            [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                        p1 = self.get_trans_point(
                            [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                        cv.line(self.base_map, tuple(p0), tuple(
                            p1), color=color, thickness=12)


if __name__ == '__main__':
    mapping = BaseSpeedLimitImgRenderer("san_mateo")
    # using cv.imwrite to .png so we can simply use cv.imread and get the exactly same matrix
    cv.imwrite(mapping.region + "_speedlimit.png", mapping.base_map)
