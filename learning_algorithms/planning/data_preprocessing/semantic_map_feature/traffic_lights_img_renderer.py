#!/usr/bin/env python

import numpy as np
import cv2 as cv

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.map.proto import map_signal_pb2
from modules.map.proto import map_overlap_pb2
from modules.perception.proto import traffic_light_detection_pb2
class TrafficLightsImgRenderer(object):
    """class of TrafficLightsImgRenderer to create images of surrounding traffic conditions"""

    def __init__(self, region):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image

        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) / 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.center = None
        self.center_heading = None

        self.region = region
        self.hd_map = None
        self.signal_dict = {}
        self.lane_dict = {}
        self.overlap_dict = {}
        self._read_hdmap()
        self._load_traffic_light()
        self._load_lane()
        self._load_overlap()

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

    def _load_traffic_light(self):
        for signal in self.hd_map.signal:
            self.signal_dict[signal.id] = signal

    def _load_lane(self):
        for lane in self.hd_map.lane:
            self.lane_dict[lane.id] = lane

    def _load_overlap(self):
        for overlap in self.hd_map.overlap:
            self.overlap_dict[overlap.id] = overlap

    def _get_traffic_light_by_id(self, id):
        return self.signal_dict[id]

    def _get_lane_by_id(self, id):
        return self.lane_dict[id]

    def _get_overlap_by_id(self, id):
        return self.overlap_dict[id]

    def _get_affine_points(self, p):
        p = p - self.center
        theta = np.pi / 2 - self.center_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_traffic_lights(self, center_x, center_y, center_heading, observed_traffic_lights):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.center = np.array([center_x, center_y])
        self.center_heading = center_heading

        for traffic_light_status in observed_traffic_lights:
            traffic_light = self._get_traffic_light_by_id(traffic_light_status.id)
            traffic_light_color = (255)
            if traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.RED:
                traffic_light_color = (255)
            elif traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.Yellow:
                traffic_light_color = (192)
            else:
                traffic_light_color = (128)

            for overlap_id in traffic_light.overlap_id:
                overlap = self._get_overlap_by_id(overlap_id)
                for overlap_object in overlap.object:
                    if overlap_object.HasField("lane_overlap_info"):
                        lane = self._get_lane_by_id(overlap_object.id)
                        for segment in lane.central_curve.segment:
                            for i in range(len(segment.line_segment.point)-1):
                                p0 = self._get_affine_points(
                                    np.array([segment.line_segment.point[i].x, segment.line_segment.point[i].y]))
                                p1 = self._get_affine_points(
                                    np.array([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y]))
                                cv.line(local_map, tuple(p0), tuple(p1),
                                        color=traffic_light_color, thickness=4)
        return local_map
                    

        
