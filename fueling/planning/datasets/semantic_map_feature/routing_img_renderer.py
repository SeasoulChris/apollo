#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv
import math

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.planning.proto import learning_data_pb2


class RoutingImgRenderer(object):
    """class of RoutingImgRenderer to create a image of routing of ego vehicle"""

    def __init__(self, region):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # TODO (Jinyun): try use a lane sequence by a fix length
        self.lane_sequence_piece_num = 5

        # lower center point in the image
        self.local_base_point_w_idx = (self.local_size_w - 1) // 2
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.center = None
        self.center_heading = None

        self.region = region
        self.hd_map = None
        self.lane_dict = {}
        self.lane_list = []
        self._read_hdmap()
        self._load_lane()

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

    def _load_lane(self):
        for lane in self.hd_map.lane:
            self.lane_dict[lane.id.id] = lane

    def _load_routing_response(self, routing_response):
        for lane_id in routing_response:
            self.lane_list.append(lane_id)

    def _calc_euclidean_dist(self, p0_x, p0_y, p1_x, p1_y):
        return math.sqrt((p0_x - p1_x)**2 + (p0_y - p1_y)**2)

    def _get_lane_sequence_by_startlane(self, start_lane_idx):
        lane_sequence = []
        lane_sequence_id = []
        for i in range(start_lane_idx, start_lane_idx + self.lane_sequence_piece_num, 1):
            if i >= len(self.lane_list):
                break
            lane_sequence.append(self.lane_dict[self.lane_list[i]])
            lane_sequence_id.append(self.lane_list[i])
        return lane_sequence

    def _get_nearest_routing_lanes(self, center_x, center_y):
        # TODO (Jinyun): use kdtree and add memory to neglect traversed lanes
        rough_distance_filter = 250  # meters
        min_distance = 50
        max_acceptable_dist = 1
        nearest_lane_id = self.lane_list[0]
        for lane_id in self.lane_list:
            lane = self.lane_dict[lane_id]
            dist_to_start = self._calc_euclidean_dist(lane.central_curve.segment[0].line_segment.point[0].x,
                                                      lane.central_curve.segment[0].line_segment.point[0].y,
                                                      center_x, center_y) > rough_distance_filter
            dist_to_end = self._calc_euclidean_dist(lane.central_curve.segment[-1].line_segment.point[-1].x,
                                                    lane.central_curve.segment[-1].line_segment.point[-1].y,
                                                    center_x, center_y) > rough_distance_filter
            if dist_to_start > rough_distance_filter and dist_to_end > rough_distance_filter:
                continue
            else:
                for segment in lane.central_curve.segment:
                    for point in segment.line_segment.point:
                        dist_to_point = self._calc_euclidean_dist(
                            point.x, point.y, center_x, center_y)
                        if dist_to_point < max_acceptable_dist:
                            return self._get_lane_sequence_by_startlane(self.lane_list.index(lane_id))
                        else:
                            if dist_to_point < min_distance:
                                nearest_lane_id = lane_id
                                min_distance = dist_to_point
        return self._get_lane_sequence_by_startlane(self.lane_list.index(nearest_lane_id))

    def _get_affine_points(self, p):
        p = p - self.center
        theta = np.pi / 2 - self.center_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    # TODO(Jinyun): to be deprecated
    def draw_routing(self, center_x, center_y, center_heading, routing_response):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.center = np.array([center_x, center_y])
        self.center_heading = center_heading
        self._load_routing_response(routing_response)

        nearest_routing_lanes = self._get_nearest_routing_lanes(
            center_x, center_y)

        routing_color_delta = int(255 / self.lane_sequence_piece_num)

        for i in range(0, self.lane_sequence_piece_num, 1):
            color = int(255 - i * routing_color_delta)
            routing_lane = nearest_routing_lanes[i]
            for segment in routing_lane.central_curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self._get_affine_points(
                        np.array([segment.line_segment.point[i].x, segment.line_segment.point[i].y]))
                    p1 = self._get_affine_points(
                        np.array([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y]))
                    cv.line(local_map, tuple(p0), tuple(
                        p1), color=color, thickness=12)
        return local_map

    def draw_local_routing(self, center_x, center_y, center_heading, local_routing):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.center = np.array([center_x, center_y])
        self.center_heading = center_heading
        routing_color_delta = int(255 / len(local_routing))
        for i in range(len(local_routing)):
            color = int(255 - i * routing_color_delta)
            routing_lane = self.lane_dict[local_routing[i]]
            for segment in routing_lane.central_curve.segment:
                for i in range(len(segment.line_segment.point)-1):
                    p0 = self._get_affine_points(
                        np.array([segment.line_segment.point[i].x, segment.line_segment.point[i].y]))
                    p1 = self._get_affine_points(
                        np.array([segment.line_segment.point[i+1].x, segment.line_segment.point[i+1].y]))
                    cv.line(local_map, tuple(p0), tuple(
                        p1), color=color, thickness=12)
        return local_map


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_local_routing/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    routing_mapping = RoutingImgRenderer("sunnyvale_with_two_offices")
    for frame in offline_frames.learning_data:
        img = routing_mapping.draw_local_routing(
            frame.localization.position.x, frame.localization.position.y,
            frame.localization.heading, frame.routing.local_routing_lane_id)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
