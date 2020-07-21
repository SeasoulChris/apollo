#!/usr/bin/env python

import threading
import math

import keyboard
import numpy as np
from torchvision import transforms

from cyber.python.cyber_py3 import cyber, cyber_time
from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_pb2

from fueling.common.coord_utils import CoordUtils
import fueling.common.logging as logging
from fueling.planning.datasets.semantic_map_feature.chauffeur_net_feature_generator \
    import ChauffeurNetFeatureGenerator
from fueling.learning.network_utils import generate_lstm_states
from fueling.planning.reinforcement_learning.rl_math_util import NormalizeAngle


class ADSEnv(object):
    def __init__(self, hidden_size=128,
                 renderer_config_file='/fuel/fueling/planning/datasets/semantic_map_feature'
                 '/planning_semantic_map_config.pb.txt',
                 region='sunnyvale_with_two_offices',
                 base_map_img_dir='/fuel/testdata/planning/semantic_map_features',
                 map_data_path="/apollo/modules/map/data/sunnyvale_with_two_officesbase_map.bin"):
        self.birdview_feature_renderer = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                      base_map_img_dir,
                                                                      region, map_data_path)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.birdview_feature_input = None

        # TODO(Songyang): specify hidden_state_size and point histpry len here or somewhere
        self.hidden = generate_lstm_states()
        self.history_point_num = 10
        self.delta_t = 0.2
        self.state = None
        self.current_adv_pose = None

        # for reward and done in function step
        self.reward = 0
        self.violation_rule = False
        self.arrival = False
        self.speed = 0
        self.is_grading_done = False
        self.is_input_ready = False

        cyber.init()
        rl_node = cyber.Node("rl_node")
        gradingsub = rl_node.create_reader("/apollo/grading",
                                           grading_result.FrameResult, self.callback_grading)
        chassissub = rl_node.create_reader("/apollo/canbus/chassis",
                                           chassis.Chassis, self.callback_chassis)
        learning_data_sub = rl_node.create_reader(
            "/apollo/planning/learning_data",
            learning_data_pb2.LearningData,
            self.callback_learning_data)

    def step(self, action):
        """
        output: next_state, reward, done, info
        done: an indicator denotes whether this episode is finished
        info: debug information
        """
        self.is_grading_done = False
        self.is_input_ready = False
        # key input: space
        keyboard.press_and_release('space')
        # send planning msg (action)
        writer = rl_node.crete_writter(
            "/apollo/planning", planning_pb2.ADCTrajectory)
        planning_message = planning_pb2.ADCTrajectory()
        planning_message.header.timestamp_sec = cyber_time.Time.now().to_sec()
        planning_message.header.module_name = "planning"
        planning_message.total_path_time = 2

        current_x, current_y, current_theta = self.current_adv_pose
        point_relative_time = 0.0
        # action in shape of [time_horizon, [dx, dy, dtheta, v]]
        for i in range(action.shape[0]):
            point = planning_message.trajectory_point.add()
            dx = action[i][0]
            dy = action[i][1]
            dtheta = action[i][2]
            point.path_point.x = dx * math.cos(current_theta) - \
                dy * math.sin(current_theta) + current_x
            point.path_point.y = dx * math.sin(current_theta) + \
                dy * math.cos(current_theta) + current_y
            point.path_point.theta = NormalizeAngle(current_theta + dtheta)
            point.v = action[i][3]
            point.relative_time = point_relative_time
            point_relative_time += self.delta_t

        accumulated_s = 0.0
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            accumulated_s += math.sqrt((nextpoint.x - point.x)
                                       ** 2 + (nextpoint.y - point.y) ** 2)
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.a = (nextpoint.v - point.v) / self.delta_t
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.da = (nextpoint.a - point.a) / self.delta_t
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.kappa = (nextpoint.theta - point.theta) / \
                (nextpoint.s - point.s)
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.dkappa = (nextpoint.kappa - point.kappa) / \
                (nextpoint.s - point.s)
        for i in range(0, len(planning_message.trajectory_point)):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.ddkappa = (nextpoint.dkappa - point.dkappa) / \
                (nextpoint.s - point.s)

        writer.write(planning_message)

        while not self.is_grading_done or not self.is_input_ready:
            time.sleep(0.1)  # second

        if self.arrival or self.violation_rule or self.collision:
            done = True
        else:
            done = False

        # Add debug information to info when in demand
        info = None

        return self.state, self.reward, done, info

    def reset(self):
        self.close()
        self.__init__()
        return self.state, self.hidden

    def close(self):
        # key input: q
        keyboard.press_and_release('q')
        pass

    def callback_grading(self, entity):
        self.reward = 0
        # the lane width is defined according to the standard freeway
        self.width_lane = 3.6

        for result in entity.detailed_result:
            if result.name == "Collision" and result.is_pass is False:
                self.reward -= 500
                self.collision = True
            if result.name == "DistanceToLaneCenter":
                dist_lane_center = result.score

            # test whether the vehicle violates the traffic rule
            if result.name == "AccelerationLimit" and result.is_pass is False:
                self.violation_rule = True
            if result.name == "RunRedLight" and result.is_pass is False:
                self.violation_rule = True
            if result.name == "SpeedLimit" and result.is_pass is False:
                self.violation_rule = True
            if result.name == "ChangeLaneAtJunction" and result.is_pass is False:
                self.violation_rule = True
            if result.name == "CrosswalkYieldToPedestrians" and result.is_pass is False:
                self.violation_rule = True
            if result.name == "RunStopSign" and result.is_pass is False:
                self.violation_rule = True

            if result.name == "DistanceToEnd":
                dist_end = result.score

        if dist_lane_center >= 0.75 * self.width_lane:
            self.reward -= 250
        if self.violation_rule:
            self.reward -= 250
        # Note: check this threshold (dist_end) in experiment
        if dist_end <= 10:
            self.reward += 100
            self.arrival = True
        self.reward -= 0.1 * dist_lane_center
        self.reward += 0.1 * self.speed
        self.is_grading_done = True

    def callback_chassis(self, entity):
        self.speed = entity.speed_mps

    def callback_learning_data(self, entity):
        frames_num = len(entity.learning_data.learning_data)
        if frames_num != 1:
            logging.info(
                "learning_data_frame's size is {}, not accepted by renderer".format(frames_num))

        frame = entity.learning_data.learning_data[0]
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        birdview_feature_input = self.img_transform(
            self.birdview_feature_renderer.
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light))
        ref_coords = [current_x,
                      current_y,
                      current_theta]

        hist_points = np.zeros((0, 4))
        for i, hist_point in enumerate(reversed(frame.adc_trajectory_point)):
            if i + 1 > self.history_point_num:
                break
            hist_x = hist_point.trajectory_point.path_point.x
            hist_y = hist_point.trajectory_point.path_point.y
            hist_theta = hist_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [hist_x, hist_y], ref_coords)
            heading_diff = hist_theta - ref_coords[2]
            hist_v = hist_point.trajectory_point.v
            hist_points = np.vstack((np.asarray(
                [local_coords[0], local_coords[1], heading_diff, hist_v]), hist_points))

        hist_points_step = np.zeros_like(hist_points)
        hist_points_step[1:, :] = hist_points[1:, :] - hist_points[:-1, :]

        self.state = tuple(
            [birdview_feature_input, hist_points, hist_points_step])
        self.current_adv_pose = tuple(ref_coords)
        self.is_input_ready = True
