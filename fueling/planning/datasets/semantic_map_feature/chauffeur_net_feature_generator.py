#!/usr/bin/env python
import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2

from fueling.planning.datasets.semantic_map_feature.agent_box_img_renderer \
    import AgentBoxImgRenderer
from fueling.planning.datasets.semantic_map_feature.agent_poses_future_img_renderer \
    import AgentPosesFutureImgRenderer
from fueling.planning.datasets.semantic_map_feature.agent_poses_history_img_renderer \
    import AgentPosesHistoryImgRenderer
from fueling.planning.datasets.semantic_map_feature.base_offroad_mask_img_renderer \
    import BaseOffroadMaskImgRenderer
from fueling.planning.datasets.semantic_map_feature.base_roadmap_img_renderer \
    import BaseRoadMapImgRenderer
from fueling.planning.datasets.semantic_map_feature.base_speedlimit_img_renderer \
    import BaseSpeedLimitImgRenderer
from fueling.planning.datasets.semantic_map_feature.obstacle_history_img_renderer \
    import ObstacleHistoryImgRenderer
from fueling.planning.datasets.semantic_map_feature.obstacle_predictions_img_renderer \
    import ObstaclePredictionsImgRenderer
from fueling.planning.datasets.semantic_map_feature.offroad_mask_img_renderer \
    import OffroadMaskImgRenderer
from fueling.planning.datasets.semantic_map_feature.roadmap_img_renderer \
    import RoadMapImgRenderer
from fueling.planning.datasets.semantic_map_feature.routing_img_renderer \
    import RoutingImgRenderer
from fueling.planning.datasets.semantic_map_feature.speed_limit_img_renderer \
    import SpeedLimitImgRenderer
from fueling.planning.datasets.semantic_map_feature.traffic_lights_img_renderer \
    import TrafficLightsImgRenderer
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


class ChauffeurNetFeatureGenerator(object):
    """class of ChauffeurNetFeatureGenerator to initialize renderers and aggregate input features"""

    def __init__(self, config_file, imgs_dir, region, map_path, base_map_update_flag=True):
        self.imgs_dir = imgs_dir
        self.config_file = config_file
        if not os.path.isfile(os.path.join(self.imgs_dir, region + ".png")) \
                or not os.path.isfile(os.path.join(self.imgs_dir, region + "_speedlimit.png")) \
                or not os.path.isfile(os.path.join(self.imgs_dir, region + "_offroad_mask.png")) \
                or base_map_update_flag:
            self.draw_base_map(self.config_file, region, map_path)
        self.agent_box_mapping = AgentBoxImgRenderer(self.config_file)
        self.agent_pose_future_mapping = AgentPosesFutureImgRenderer(
            self.config_file)
        self.agent_pose_history_mapping = AgentPosesHistoryImgRenderer(
            self.config_file)
        self.obstacle_history_mapping = ObstacleHistoryImgRenderer(
            self.config_file)
        self.obstacle_predictions_mapping = ObstaclePredictionsImgRenderer(
            self.config_file)
        self.offroad_mask_mapping = OffroadMaskImgRenderer(
            self.config_file, region, imgs_dir)
        self.road_map_mapping = RoadMapImgRenderer(self.config_file, region, imgs_dir)
        self.routing_mapping = RoutingImgRenderer(self.config_file, region, map_path)
        self.speed_limit_mapping = SpeedLimitImgRenderer(
            self.config_file, region, imgs_dir)
        self.traffic_lights_mapping = TrafficLightsImgRenderer(
            self.config_file, region, map_path)

    def draw_base_map(self, config_file, region, map_path):
        self.base_offroad_mask_mapping = BaseOffroadMaskImgRenderer(
            config_file, region, map_path)
        self.base_road_map_mapping = BaseRoadMapImgRenderer(
            config_file, region, map_path)
        self.base_speed_limit_mapping = BaseSpeedLimitImgRenderer(
            config_file, region, map_path)
        cv.imwrite(os.path.join(self.imgs_dir, region + "_offroad_mask.png"),
                   self.base_offroad_mask_mapping.base_map)
        cv.imwrite(os.path.join(self.imgs_dir, region + ".png"),
                   self.base_road_map_mapping.base_map)
        cv.imwrite(os.path.join(self.imgs_dir, region + "_speedlimit.png"),
                   self.base_speed_limit_mapping.base_map)

    def render_seperated_img_features(self, frame_num, frame_time_sec, ego_pose_history,
                                      obstacle, center_x, center_y, center_heading,
                                      routing_response, observed_traffic_lights, output_dirs,
                                      coordinate_heading=0., past_motion_dropout=False):
        '''
        For debug purposes, features are drawn seprately
        agent_box_img: 1 channel np.array image
        agent_pose_history_img: 1 channel np.array image
        obstacle_history_img: 1 channel np.array image
        obstacle_predictions_img: 1 channel np.array image
        road_map_img: 3 channel np.array image
        routing_img: 1 channel np.array image
        speed_limit_img: 3 channel np.array image
        traffic_lights_img: 1 channel np.array image
        '''
        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            frame_time_sec, center_x, center_y, center_heading,
            ego_pose_history, coordinate_heading, past_motion_dropout)
        obstacle_history_img = self.obstacle_history_mapping.draw_obstacle_history(
            frame_time_sec, obstacle, coordinate_heading)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_box_prediction(
            frame_time_sec, obstacle, coordinate_heading)
        road_map_img = self.road_map_mapping.draw_roadmap(
            center_x, center_y, center_heading, coordinate_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            center_x, center_y, center_heading, routing_response, coordinate_heading)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            center_x, center_y, center_heading, coordinate_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            center_x, center_y, center_heading, observed_traffic_lights, coordinate_heading)
        imgs_list = [agent_box_img, agent_pose_history_img,
                     obstacle_history_img, obstacle_predictions_img,
                     road_map_img, routing_img, speed_limit_img, traffic_lights_img]
        for i in range(len(output_dirs)):
            key = "{}@{:.3f}".format(frame_num, frame_time_sec)
            filename = key + ".png"
            cv.imwrite(os.path.join(output_dirs[i], filename), imgs_list[i])

    def render_stacked_img_features(self, frame_num, frame_time_sec, ego_pose_history,
                                    obstacle, center_x, center_y, center_heading,
                                    routing_response, observed_traffic_lights,
                                    coordinate_heading=0., past_motion_dropout=False):
        '''
        agent_box_img: 1 channel np.array image, [0]
        agent_pose_history_img: 1 channel np.array image, [1]
        obstacle_history_img: 1 channel np.array image, [2]
        obstacle_predictions_img: 1 channel np.array image, [3]
        road_map_img: 3 channel np.array image, [4:7]
        routing_img: 1 channel np.array image, [7]
        speed_limit_img: 3 channel np.array image, [8:11]
        traffic_lights_img: 1 channel np.array image, [11]

        All images in np.unit8 and concatenated along channel axis
        '''

        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            frame_time_sec, center_x, center_y, center_heading, ego_pose_history,
            coordinate_heading, past_motion_dropout)
        obstacle_history_img = self.obstacle_history_mapping.draw_obstacle_history(
            frame_time_sec, obstacle, coordinate_heading)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_box_prediction(
            frame_time_sec, obstacle, coordinate_heading)
        road_map_img = self.road_map_mapping.draw_roadmap(
            center_x, center_y, center_heading, coordinate_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            center_x, center_y, center_heading, routing_response, coordinate_heading)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            center_x, center_y, center_heading, coordinate_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            center_x, center_y, center_heading, observed_traffic_lights, coordinate_heading)

        return np.concatenate([agent_box_img, agent_pose_history_img,
                               obstacle_history_img, obstacle_predictions_img,
                               road_map_img, routing_img, speed_limit_img,
                               traffic_lights_img], axis=2)

    def render_merged_img_feature(self, stacked_img_features):
        '''
        Merging sequence would be as follow up to down:
        road_map_img: 3 channel np.array image
        routing_img: 1 channel np.array image
        speed_limit_img: 3 channel np.array image
        traffic_lights_img: 1 channel np.array image
        obstacle_history_img: 1 channel np.array image
        obstacle_predictions_img: 1 channel np.array image
        agent_box_img: 1 channel np.array image
        agent_pose_history_img: 1 channel np.array image
        '''
        road_map_img = stacked_img_features[:, :, 4:7]
        routing_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 7], axis=2), 3, axis=2)
        # speed_limit_img = stacked_img_features[:, :, 8:11]
        traffic_lights_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 11], 2), 3, axis=2)
        # draw obstacle past in red color
        obstacle_history_img = renderer_utils.img_white_gradient_to_color_gradient(
            np.repeat(np.expand_dims(
                stacked_img_features[:, :, 2],
                axis=2),
                3, axis=2),
            (0, 0, 255))
        # draw obstacle future in green color
        obstacle_predictions_img = renderer_utils.img_white_gradient_to_color_gradient(
            np.repeat(np.expand_dims(
                stacked_img_features[:, :, 3],
                axis=2),
                3, axis=2),
            (0, 255, 0))
        agent_box_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 0], axis=2), 3, axis=2)
        agent_pose_history_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 1], axis=2), 3, axis=2)
        merged_img = renderer_utils.img_notblack_stacking(
            routing_img, road_map_img)
        # merged_img = renderer_utils.img_notblack_stacking(
        #     speed_limit_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            traffic_lights_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            obstacle_history_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            obstacle_predictions_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            agent_box_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            agent_pose_history_img, merged_img)
        return merged_img

    def render_gt_pose_dist(self, center_x, center_y, center_heading,
                            ego_pose_future, timestamp_idx, coordinate_heading=0.):
        return self.agent_pose_future_mapping.draw_agent_pose_future(center_x,
                                                                     center_y,
                                                                     center_heading,
                                                                     ego_pose_future,
                                                                     timestamp_idx,
                                                                     coordinate_heading)

    def render_gt_box(self, center_x, center_y, center_heading,
                      ego_pose_future, timestamp_idx, coordinate_heading=0.):
        return self.agent_pose_future_mapping.draw_agent_box_future(center_x,
                                                                    center_y,
                                                                    center_heading,
                                                                    ego_pose_future,
                                                                    timestamp_idx,
                                                                    coordinate_heading)

    def render_offroad_mask(self, center_x, center_y, center_heading,
                            coordinate_heading=0.):
        return self.offroad_mask_mapping.draw_offroad_mask(center_x,
                                                           center_y,
                                                           center_heading,
                                                           coordinate_heading)

    def render_obstacle_box_prediction_frame(
            self, center_x, center_y, center_heading,
            obstacles, timestamp_idx, coordinate_heading=0.):
        return self.obstacle_predictions_mapping.draw_obstacle_box_prediction_frame(
            obstacles,
            timestamp_idx,
            coordinate_heading)

    def render_initial_agent_states(self, coordinate_heading=0.):
        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_img = self.agent_pose_history_mapping.draw_agent_current_pose()
        return agent_box_img, agent_pose_img


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/output_data_evaluated/test/2019-10-17-13-36-41/complete/"
              "test_set/00006.record.49.bin.future_status.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    region = "sunnyvale_with_two_offices"
    config_file = "/fuel/fueling/planning/datasets/semantic_map_feature/" \
        "planning_semantic_map_config.pb.txt"
    imgs_dir = '/fuel/testdata/planning/semantic_map_features'
    chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(
        config_file,
        imgs_dir,
        region, base_map_update_flag=False)
    print("Finish loading chauffeur_net_feature_generator...")

    imgs_dir = '/fuel/testdata/planning/semantic_map_features'
    output_dirs = [os.path.join(imgs_dir, 'data_local_agent_box/'),
                   os.path.join(imgs_dir, 'data_agent_pose_history/'),
                   os.path.join(imgs_dir, 'data_obstacles/'),
                   os.path.join(imgs_dir, 'data_obstacle_predictions/'),
                   os.path.join(imgs_dir, 'data_local_road_map/'),
                   os.path.join(imgs_dir, 'data_local_routing/'),
                   os.path.join(imgs_dir, 'data_local_speed_limit/'),
                   os.path.join(imgs_dir, 'data_traffic_light/')]
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print("Making output directory: " + output_dir)

    for frame in offline_frames.learning_data_frame:
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta
        chauffeur_net_feature_generator.render_seperated_img_features(
            frame.frame_num,
            frame.adc_trajectory_point[-1].timestamp_sec,
            frame.adc_trajectory_point,
            frame.obstacle,
            current_x,
            current_y,
            current_theta,
            frame.routing.local_routing_lane_id,
            frame.traffic_light_detection.traffic_light,
            output_dirs)
