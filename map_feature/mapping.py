###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import numpy as np
import cv2 as cv
from modules.map.proto import map_pb2
from scipy import misc


class Mapping(object):
    """class of Mapping to get a feature map"""
    def __init__(self, region):
        """contruct function to init Mapping object"""
        self.region = region
        if (self.region == "san_mateo"):
            self.GRID = [10000, 12000]
            self.base_point = np.array([559000, 4156860])
            self.resolution = 0.1
        if (self.region == "sunnyvale_with_two_offices"):
            self.GRID = [26000, 18000]
            self.base_point = np.array([585950, 4140000])
            self.resolution = 0.1

        self.base_map = np.zeros([self.GRID[1], self.GRID[0], 3], dtype=np.uint8)
        self._read_hdmap()
        self._draw_base_map()

            
    def _read_hdmap(self):
        """read the hdmap from base_map.bin"""
        self.hd_map = map_pb2.Map()
        map_path = "/apollo/modules/map/data/" + self.region + "/base_map.bin"
        print("Loading map from file: " + map_path)
        with open(map_path, 'rb') as file_in:
            self.hd_map.ParseFromString(file_in.read())
        # p_min, p_max = self._get_map_base_point()
        # print(p_min, p_max)

    def _draw_base_map(self):
        self._draw_road()
        self._draw_junction()
        self._draw_crosswalk()
        self._draw_lane_boundary()
        self._draw_lane_central()


    def get_trans_point(self, p):
        point = np.round((p - self.base_point) / self.resolution)
        return [int(point[0]), self.GRID[1] - int(point[1])]

    def _update_base_point(self, line_segment, p_min, p_max):
        for point in line_segment.point:
            p_min = np.minimum(p_min, [point.x, point.y])
            p_max = np.maximum(p_max, [point.x, point.y])
        return p_min, p_max

    def _get_map_base_point(self):
        """get hd_map base points"""
        p_min = np.array([0, 0])
        p_max = np.array([0, 0])
        p_min[0] = self.hd_map.lane[0].left_boundary.curve.segment[0].line_segment.point[0].x
        p_min[1] = self.hd_map.lane[0].left_boundary.curve.segment[0].line_segment.point[0].y
        p_max[0] = self.hd_map.lane[0].left_boundary.curve.segment[0].line_segment.point[0].x
        p_max[1] = self.hd_map.lane[0].left_boundary.curve.segment[0].line_segment.point[0].y

        for lane in self.hd_map.lane:
            for segment in lane.left_boundary.curve.segment:
                if segment.HasField('line_segment'):
                    (p_min, p_max) = self._update_base_point(segment.line_segment, p_min, p_max)

            for segment in lane.right_boundary.curve.segment:
                if segment.HasField('line_segment'):
                    (p_min, p_max) = self._update_base_point(segment.line_segment, p_min, p_max)
                    
        for crosswalk in self.hd_map.crosswalk:
            if crosswalk.HasField('polygon'):
                (p_min, p_max) = self._update_base_point(crosswalk.polygon, p_min, p_max)

        for road in self.hd_map.road:
            for section in road.section:
                for edge in section.boundary.outer_polygon.edge:
                    for segment in edge.curve.segment:
                        if segment.HasField('line_segment'):
                            (p_min, p_max) = self._update_base_point(segment.line_segment, p_min, p_max)
        return p_min, p_max

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
        hsv_dict = {0 : (V, K, M), 
                    1 : (N, V, M),
                    2 : (M, V, K),
                    3 : (M, N, V),
                    4 : (K, M, V),
                    5 : (V, M, N)}
        return tuple(x*255 for x in hsv_dict[I])

    def _draw_road(self, color=(64,64,64)):
        for road in self.hd_map.road:
            for section in road.section:
                points = np.zeros((0,2))
                for edge in section.boundary.outer_polygon.edge:
                    if edge.type == 2:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point)):
                                point = self.get_trans_point([segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                                points = np.vstack((points, point))
                    elif edge.type == 3:
                        for segment in edge.curve.segment:
                            for i in range(len(segment.line_segment.point)-1, -1, -1):
                                point = self.get_trans_point([segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                                points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_junction(self, color=(128,128,128)):
        for junction in self.hd_map.junction:
            if junction.HasField("polygon"):
                points = np.zeros((0,2))
                for i in range(len(junction.polygon.point)):
                    point = self.get_trans_point([junction.polygon.point[i].x, junction.polygon.point[i].y])
                    points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_crosswalk(self, color=(192,192,192)):
        for crosswalk in self.hd_map.crosswalk:
            if crosswalk.HasField("polygon"):
                points = np.zeros((0,2))
                for i in range(len(crosswalk.polygon.point)):
                    point = self.get_trans_point([crosswalk.polygon.point[i].x, crosswalk.polygon.point[i].y])
                    points = np.vstack((points, point))
                cv.fillPoly(self.base_map, [np.int32(points)], color=color)

    def _draw_lane_boundary(self, color=(255,255,255)):
        for lane in self.hd_map.lane:
            if lane.type == 2 and lane.left_boundary.virtual and lane.right_boundary.virtual:
                continue
            for segment in lane.left_boundary.curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point([segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    cv.line(self.base_map, tuple(p0), tuple(p1), color=color, thickness = 2)
            for segment in lane.right_boundary.curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point([segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    cv.line(self.base_map, tuple(p0), tuple(p1), color=color, thickness = 2)

    def _draw_lane_central(self):
        for lane in self.hd_map.lane:
            for segment in lane.central_curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self.get_trans_point([segment.line_segment.point[i].x, segment.line_segment.point[i].y])
                    p1 = self.get_trans_point([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y])
                    theta = np.arctan2(segment.line_segment.point[i+1].y-segment.line_segment.point[i].y, segment.line_segment.point[i+1].x-segment.line_segment.point[i].x)/(2*np.pi) % 1
                    cv.line(self.base_map, tuple(p0), tuple(p1), color=self._hsv_to_rgb(theta), thickness = 4)

if __name__ == '__main__':
    mapping = Mapping("san_mateo")
    # using cv.imwrite to .png so we can simply use cv.imread and get the exactly same matrix
    cv.imwrite(mapping.region + ".png", mapping.base_map)
    # misc.imsave(self.region + ".jpg", self.base_map)