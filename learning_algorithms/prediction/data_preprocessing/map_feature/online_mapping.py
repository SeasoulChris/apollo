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

import os
import shutil
import numpy as np
import cv2 as cv
# from learning_algorithms.prediction.data_preprocessing.map_feature.mapping import Mapping


class ObstacleMapping(object):
    """class of ObstacleMapping to create an obstacle feature_map"""
    def __init__(self, region, world_coord, obstacles_history):
        """contruct function to init ObstacleMapping object"""
        center_point = world_coord[0:2]
        if region=="san_mateo" and os.path.exists("san_mateo.png"):
            base_map = cv.imread("san_mateo.png")
            # print("san_mateo.png exist, read it directly!")
            center_idx = [int(np.round((center_point[0]-558980)/0.1)), int(14000-np.round((center_point[1]-4156780)/0.1))]
        elif region=="sunnyvale_with_two_offices" and os.path.exists("sunnyvale_with_two_offices.png"):
            base_map = cv.imread("sunnyvale_with_two_offices.png")
            # print("sunnyvale_with_two_offices.png exist, read it directly!")
            center_idx = [int(np.round((center_point[0]-585870)/0.1)), int(20000-np.round((center_point[1]-4139900)/0.1))]
        else:
            mapping = Mapping(region)
            base_map = mapping.base_map
            cv.imwrite(mapping.region + ".png", base_map)
            print("Drawing map: " + mapping.region + ".png")
            center_idx = mapping.get_trans_point(center_point)

        self.world_coord = world_coord
        self.obstacles_history = obstacles_history
        self.base_point = np.array(center_point) - 100
        self.GRID = [2000, 2000]
        self.resolution = 0.1
        self.feature_map = base_map[center_idx[1]-1000:center_idx[1]+1000, center_idx[0]-1000:center_idx[0]+1000]
        self.draw_obstacles_history()

    def get_trans_point(self, p):
        point = np.round((p - self.base_point) / self.resolution)
        return [int(point[0]), self.GRID[1] - int(point[1])]

    def _draw_polygon(self, feature_map, polygon_points, color=(0,255,255)):
        points = np.zeros((0,2))
        for polygon_point in polygon_points:
            if (polygon_point==np.array([0,0])).all():
                break
            point = self.get_trans_point(polygon_point)
            points = np.vstack((points, point))
        cv.fillPoly(feature_map, [np.int32(points)], color=color)

    def _draw_rectangle(self, feature_map, pos_point, color=(0,255,255)):
        w, l = 2.11, 4.93
        theta = self.world_coord[2]
        points = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), \
                        np.array([[l/2, l/2, -l/2, -l/2], [w/2, -w/2, -w/2, w/2]])).T + np.array(pos_point)
        points = [self.get_trans_point(point) for point in points]
        cv.fillPoly(feature_map, [np.int32(points)], color=color)

    def draw_history(self, feature_map, history, color=(0,255,255)):
        # draw obstacle_history
        # history is a np.array with shape [history_size, polygon_point_size, 2]
        history_size = history.shape[0]
        for i in range(history_size-10, history_size):
            polygon_points = history[i]
            if (polygon_points==np.zeros([20,2])).all():
                continue
            self._draw_polygon(feature_map, polygon_points, tuple(c*(1/history_size*i) for c in color))
 
    def draw_obstacles_history(self, color=(0,255,255)):
        for history in self.obstacles_history:
            self.draw_history(self.feature_map, history, (0,255,255))

    def crop_area(self, feature_map, world_coord):
        center = tuple(self.get_trans_point(world_coord[0:2]))
        heading_angle = world_coord[2] * 180 / np.pi
        M = cv.getRotationMatrix2D(center, 90-heading_angle, 1.0)
        rotated = cv.warpAffine(feature_map, M, tuple(self.GRID))
        output = rotated[center[1]-300:center[1]+100, center[0]-200:center[0]+200]
        return cv.resize(output, (224,224))

    def crop_by_history(self, history, color=(0,0,255)):
        feature_map = self.feature_map.copy()
        self.draw_history(feature_map, history, color)
        return self.crop_area(feature_map, self.world_coord)

