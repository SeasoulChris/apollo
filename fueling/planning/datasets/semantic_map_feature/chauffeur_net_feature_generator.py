#!/usr/bin/env python
import os
import shutil

import numpy as np
import cv2 as cv

from fueling.planning.datasets.semantic_map_feature.agent_box_img_renderer import AgentBoxImgRenderer
from fueling.planning.datasets.semantic_map_feature.agent_poses_history_img_renderer import AgentPosesHistoryImgRenderer
from fueling.planning.datasets.semantic_map_feature.base_roadmap_img_renderer import BaseRoadMapImgRenderer
from fueling.planning.datasets.semantic_map_feature.base_speedlimit_img_renderer import BaseSpeedLimitImgRenderer
from fueling.planning.datasets.semantic_map_feature.obstacles_img_renderer import ObstaclesImgRenderer
from fueling.planning.datasets.semantic_map_feature.obstacle_predictions_img_renderer import ObstaclePredictionsImgRenderer
from fueling.planning.datasets.semantic_map_feature.roadmap_img_renderer import RoadMapImgRenderer
from fueling.planning.datasets.semantic_map_feature.routing_img_renderer import RoutingImgRenderer
from fueling.planning.datasets.semantic_map_feature.speed_limit_img_renderer import SpeedLimitImgRenderer
from fueling.planning.datasets.semantic_map_feature.traffic_lights_img_renderer import TrafficLightsImgRenderer
from modules.planning.proto import learning_data_pb2


class ChauffeurNetFeatureGenerator(object):
    """class of ChauffeurNetFeatureGenerator to initialize renderers and aggregate input features"""

    def __init__(self, region):
        self.imgs_dir = "/fuel/testdata/planning/semantic_map_features"
        self.draw_base_map()
        self.agent_box_mapping = AgentBoxImgRenderer()
        self.agent_pose_history_mapping = AgentPosesHistoryImgRenderer()
        self.obstacles_mapping = ObstaclesImgRenderer()
        self.obstacle_predictions_mapping = ObstaclePredictionsImgRenderer()
        self.road_map_mapping = RoadMapImgRenderer(region)
        self.routing_mapping = RoutingImgRenderer(region)
        self.speed_limit_mapping = SpeedLimitImgRenderer(region)
        self.traffic_lights_mapping = TrafficLightsImgRenderer(region)

    def draw_base_map(self):
        self.base_road_map_mapping = BaseRoadMapImgRenderer(region)
        self.base_speed_limit_mapping = BaseSpeedLimitImgRenderer(region)
        cv.imwrite(os.path.join(self.imgs_dir, self.base_road_map_mapping.region + ".png"),
                   self.base_road_map_mapping.base_map)
        cv.imwrite(os.path.join(self.imgs_dir, self.base_speed_limit_mapping.region + "_speedlimit.png"),
                   self.base_speed_limit_mapping.base_map)

    def draw_features(self, frame_num, frame_time_sec, ego_pose_history, obstacle,
                      center_x, center_y, center_heading, routing_response,
                      observed_traffic_lights, output_dirs):
        agent_box_img = self.agent_box_mapping.draw_agent_box()
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            ego_pose_history)
        obstacles_img = self.obstacles_mapping.draw_obstacles(
            frame_time_sec, obstacle)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_prediction(
            center_x, center_y, center_heading, obstacle)
        road_map_img = self.road_map_mapping.draw_roadmap(
            center_x, center_y, center_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            center_x, center_y, center_heading, routing_response)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            center_x, center_y, center_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            center_x, center_y, center_heading, observed_traffic_lights)
        imgs_list = [agent_box_img, agent_pose_history_img, obstacles_img, obstacle_predictions_img,
                     road_map_img, routing_img, speed_limit_img, traffic_lights_img]
        for i in range(len(output_dirs)):
            key = "{}@{:.3f}".format(frame_num, frame_time_sec)
            filename = key + ".png"
            cv.imwrite(os.path.join(output_dirs[i], filename), imgs_list[i])

    def render_stacked_img_features(self, frame_num, frame_time_sec, ego_pose_history, obstacle,
                                    center_x, center_y, center_heading, routing_response,
                                    observed_traffic_lights):
        agent_box_img = self.agent_box_mapping.draw_agent_box()
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            ego_pose_history)
        obstacles_img = self.obstacles_mapping.draw_obstacles(
            frame_time_sec, obstacle)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_prediction(
            center_x, center_y, center_heading, obstacle)
        road_map_img = self.road_map_mapping.draw_roadmap(
            center_x, center_y, center_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            center_x, center_y, center_heading, routing_response)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            center_x, center_y, center_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            center_x, center_y, center_heading, observed_traffic_lights)
        return np.concatenate([agent_box_img, agent_pose_history_img, obstacles_img, obstacle_predictions_img,
                               road_map_img, routing_img, speed_limit_img, traffic_lights_img], axis=2)


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.31.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    region = "sunnyvale_with_two_offices"
    chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(region)
    print("Finish loading chauffeur_net_feature_generator...")

    output_dirs = ['./data_local_agent_box/', './data_agent_pose_history/', './data_obstacles/', './data_obstacle_predictions/',
                   './data_local_road_map/', './data_local_routing/', './data_local_speed_limit/', './data_traffic_light/']
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print("Making output directory: " + output_dir)

    for frame in offline_frames.learning_data:
        chauffeur_net_feature_generator.draw_features(frame.frame_num,
                                                      frame.timestamp_sec,
                                                      frame.adc_trajectory_point,
                                                      frame.obstacle,
                                                      frame.localization.position.x,
                                                      frame.localization.position.y,
                                                      frame.localization.heading,
                                                      frame.routing.local_routing_lane_id,
                                                      frame.traffic_light,
                                                      output_dirs)
        print(frame.frame_num)
