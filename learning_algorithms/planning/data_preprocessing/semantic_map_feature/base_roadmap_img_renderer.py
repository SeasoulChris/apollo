#!/usr/bin/env python

import numpy as np
import cv2 as cv
import pyproj

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2

class BaseRoadMapImgRenderer(object):
    """class of BaseRoadMapImgRenderer to get a feature map according to Baidu Apollo Map Format"""

    def __init__(self, region):
        """contruct function to init BaseRoadMapImgRenderer object"""
        self.region = region
        # TODO(Jinyun): use config file
        self.resolution = 0.1   # in meter/pixel
        self.base_map_padding = 100    # in meter

        self.base_point = None
        self.GRID = None
        self.base_map = None

        self._read_hdmap()
        self._build_canvas()
        self._draw_base_map()
        print("Base Map base point is " + str(self.base_point[0]) + ", " + str(self.base_point[1]))
        print("Base Map W * H is " + str(self.GRID[0]) + " * " + str(self.GRID[1]))

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
        self._draw_road()
        self._draw_junction()
        self._draw_crosswalk()
        self._draw_lane_boundary()
        self._draw_stop_line()
        # self._draw_lane_central()

    def get_trans_point(self, p):
        point = np.round((p - self.base_point) / self.resolution)
        return [int(point[0]), self.GRID[1] - int(point[1])]

    def _hsv_to_rgb(self, H=1.0, S=1.0, V=1.0):
        """
        convert HSV to RGB color,
        see http://www.icst.pku.edu.cn/F/course/ImageProcessing/2018/resource/Color78.pdf, page 5
        """
        H = H * 6
        I = H // 1
        F = H % 1
        M = V * (1 - S)
        N = V * (1 - S * F)
        K = V * (1 - S * (1 - F))
        hsv_dict = {0: (V, K, M),
                    1: (N, V, M),
                    2: (M, V, K),
                    3: (M, N, V),
                    4: (K, M, V),
                    5: (V, M, N)}
        return tuple(x*255 for x in hsv_dict[I])

    def _draw_road(self, color=(64, 64, 64)):
        for road in self.hd_map.road:
            for section in road.section:
                points = np.zeros((0, 2))
                for edge in section.boundary.outer_polygon.edge:
                    if edge.type == 2:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point)):
                                point = self.get_trans_point(
                                    [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                                points = np.vstack((points, point))
                    elif edge.type == 3:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point)-1, -1, -1):
                                point = self.get_trans_point(
                                    [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                                points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_junction(self, color=(128, 128, 128)):
        for junction in self.hd_map.junction:
            if junction.HasField("polygon"):
                points = np.zeros((0, 2))
                for i in range(len(junction.polygon.point)):
                    point = self.get_trans_point(
                        [junction.polygon.point[i].x, junction.polygon.point[i].y])
                    points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_crosswalk(self, color=(192, 192, 192)):
        for crosswalk in self.hd_map.crosswalk:
            if crosswalk.HasField("polygon"):
                points = np.zeros((0, 2))
                for i in range(len(crosswalk.polygon.point)):
                    point = self.get_trans_point(
                        [crosswalk.polygon.point[i].x, crosswalk.polygon.point[i].y])
                    points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_stop_line(self, color=(0, 0, 255)):
        for stop_sign in self.hd_map.stop_sign:
            for stop_line in stop_sign.stop_line:
                for segment in stop_line.segment:
                    for i in range(len(segment.line_segment.point)-1):
                        p0 = self.get_trans_point(
                            [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                        p1 = self.get_trans_point(
                            [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                        cv.line(self.base_map, tuple(p0), tuple(
                            p1), color=color, thickness=4)

    def _draw_lane_boundary(self, white_color=(255, 255, 255), yellow_color=(0, 255, 255)):
        for lane in self.hd_map.lane:
            # TODO(Jinyun): try to use dot line to present virtual lane boundary in intersection
            if lane.left_boundary.virtual and lane.right_boundary.virtual:
                continue
            color = white_color
            # TODO(Jinyun): no DOUBLE_YELLOW and CURB boundary from map file! To Implement DOTTED WHITE and DOTTED YELLOW
            if lane.left_boundary.boundary_type[0].types[0] == map_lane_pb2.LaneBoundaryType.Type.SOLID_YELLOW:
                color = yellow_color
            for segment in lane.left_boundary.curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point(
                        [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point(
                        [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    cv.line(self.base_map, tuple(p0), tuple(
                        p1), color=color, thickness=2)

            color = white_color
            if lane.right_boundary.boundary_type[0].types[0] == map_lane_pb2.LaneBoundaryType.Type.SOLID_YELLOW:
                color = yellow_color
            for segment in lane.right_boundary.curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point(
                        [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point(
                        [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    cv.line(self.base_map, tuple(p0), tuple(
                        p1), color=color, thickness=2)

    def _draw_lane_central(self):
        for lane in self.hd_map.lane:
            for segment in lane.central_curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point(
                        [segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point(
                        [segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    theta = np.arctan2(segment.line_segment.point[i+1].y-segment.line_segment.point[i].y,
                                       segment.line_segment.point[i+1].x-segment.line_segment.point[i].x)/(2*np.pi) % 1
                    cv.line(self.base_map, tuple(p0), tuple(p1),
                            color=self._hsv_to_rgb(theta), thickness=4)


if __name__ == '__main__':
    mapping = BaseRoadMapImgRenderer("san_mateo")
    # using cv.imwrite to .png so we can simply use cv.imread and get the exactly same matrix
    cv.imwrite(mapping.region + ".png", mapping.base_map)
