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
from modules.map.proto import map_pb2
from modules.prediction.proto import offline_features_pb2
from mapping import Mapping


class ObstacleMapping(object):
    """class of ObstacleMapping to create an obstacle feature_map"""
    def __init__(self, region, frame_env):
        """contruct function to init ObstacleMapping object"""
        center_point = np.array([frame_env.ego_history.feature[0].position.x, frame_env.ego_history.feature[0].position.y])
        if region=="san_mateo" and os.path.exists("san_mateo.png"):
            base_map = cv.imread("san_mateo.png")
            print("san_mateo.png exist, read it directly!")
            center_idx = [int(np.round((center_point[0]-559000)/0.1)), int(12000-np.round((center_point[1]-4156860)/0.1))]
        else:
            mapping = Mapping(region)
            base_map = mapping.base_map
            center_idx = mapping.get_trans_point(center_point)

        self.frame_env = frame_env
        self.timestamp = self.frame_env.timestamp
        self.base_point = np.array(center_point) - 100
        self.GRID = [2000, 2000]
        self.resolution = 0.1
        self.feature_map = base_map[center_idx[1]-1000:center_idx[1]+1000, center_idx[0]-1000:center_idx[0]+1000]
        self.draw_frame()

    def get_trans_point(self, p):
        point = np.round((p - self.base_point) / self.resolution)
        return [int(point[0]), self.GRID[1] - int(point[1])]

    def _draw_rectangle(self, feature_map, feature, color=(0,255,255)):
        pos_x = feature.position.x
        pos_y = feature.position.y
        w, l = feature.width, feature.length
        theta = feature.theta
        points = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), \
                        np.array([[l/2, l/2, -l/2, -l/2], [w/2, -w/2, -w/2, w/2]])).T + np.array([pos_x, pos_y])
        points = [self.get_trans_point(point) for point in points]
        cv.fillPoly(feature_map, [np.int32(points)], color=color)

    def _draw_polygon(self, feature_map, feature, color=(0,255,255)):
        points = np.zeros((0,2))
        for polygon_point in feature.polygon_point:
            point = self.get_trans_point([polygon_point.x, polygon_point.y])
            points = np.vstack((points, point))
        cv.fillPoly(feature_map, [np.int32(points)], color=color)

    def draw_history(self, feature_map, history, color=(0,255,255)):
        # draw obstacle_history in reverse order
        for i in range(len(history.feature)-1, -1, -1):
            feature = history.feature[i]
            if feature.id == -1:
                self._draw_rectangle(feature_map, feature, tuple(c*(1-self.timestamp+feature.timestamp) for c in color))
            else:
                self._draw_polygon(feature_map, feature, tuple(c*(1-self.timestamp+feature.timestamp) for c in color))

    def draw_frame(self, color=(0,255,255)):
        for history in self.frame_env.obstacles_history:
            self.draw_history(self.feature_map, history, (0,255,255))
        self.draw_history(self.feature_map, self.frame_env.ego_history, (0,255,255))

    def crop_area(self, feature_map, center_point, heading):
        center = tuple(self.get_trans_point(center_point))
        heading_angle = heading * 180 / np.pi
        M = cv.getRotationMatrix2D(center, 90-heading_angle, 1.0)
        rotated = cv.warpAffine(feature_map, M, tuple(self.GRID))
        output = rotated[center[1]-300:center[1]+100, center[0]-200:center[0]+200]
        return cv.resize(output, (224,224))

    def crop_by_history(self, history, color=(0,0,255)):
        feature_map = self.feature_map.copy()
        self.draw_history(feature_map, history, color)
        curr_feature = history.feature[0]
        return self.crop_area(feature_map, [curr_feature.position.x, curr_feature.position.y], curr_feature.theta)



if __name__ == '__main__':
    list_frame = offline_features_pb2.ListFrameEnv()
    with open("/apollo/data/frame_env.0.bin", 'r') as file_in:
        list_frame.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    obs_pos_dict = dict()
    for frame_env in list_frame.frame_env:
        obstacle_mapping = ObstacleMapping("san_mateo", frame_env)
        for history in frame_env.obstacles_history:
            if not history.is_trainable:
                continue
            key = "{}@{:.3f}".format(history.feature[0].id, history.feature[0].timestamp)
            filename = key + ".png"
            obs_pos = []
            for feature in history.feature:
                obs_pos.append((feature.position.x, feature.position.y))
            obs_pos_dict[key] = obs_pos
            img = obstacle_mapping.crop_by_history(history)
            cv.imwrite(output_dir + filename, img)
    np.save(output_dir+"obs_pos.npy", obs_pos_dict)
