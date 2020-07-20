import threading
import math

import keyboard

from fueling.learning.network_utils import generate_lstm_states
from cyber.python.cyber_py3 import cyber, cyber_time


class ADSEnv(object):
    def __init__(self, hidden_size=128):
        self.hidden = generate_lstm_states()
        self.state = self.semantic_map()

        # for reward and done in function step
        self.reward = 0
        self.violation_rule = False
        self.arrival = False
        self.speed = 0
        self.is_env_ready = False

        cyber.init()
        rl_node = cyber.Node("rl_node")
        gradingsub = rl_node.create_reader("/apollo/grading",
                                           grading_result.FrameResult, self.callback_grading)
        chassissub = rl_node.create_reader("/apollo/canbus/chassis",
                                           chassis.Chassis, self.callback_chassis)

    def step(self, action):
        """
        output: next_state, reward, done, info
        done: an indicator denotes whether this episode is finished
        info: debug information
        """
        self.is_env_ready = False
        # key input: space
        keyboard.press_and_release('space')
        # send planning msg (action)
        writer = rl_node.crete_writter("/apollo/planning", ADCTrajectory)
        planning = ADCTrajectory()
        planning.header.timestamp_sec = cyber_time.Time.now().to_sec()
        planning.header.module_name = "planning"
        planning.total_path_time = 2

        # TODO(Jinyun): check the decomposition of action
        # TODO(Jiaming): revise and add more info for the traj_point
        for dx, dy, dheading, speed in action:
            point = planning.trajectory_point.add()
            point.path_point.x = dx
            point.path_point.y = dy
            point.path_point.theta = dheading
            point.v = speed
            point.relative_time = 0.2
            plannning.trajectory_point.append(point)

        accumulated_s = 0.0
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            accumulated_s += math.sqrt((nextpoint.x - point.x) ** 2 + (nextpoint.y - point.y) ** 2)
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            point.a = (nextpoint.v - point.v) / 0.2
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            point.da = (nextpoint.a - point.a) / 0.2
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            point.kappa = nextpoint.theta - point.theta
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            point.dkappa = (nextpoint.theta - point.theta) / (nextpoint.s - point.s)
        for i in range(0, len(planning.trajectory_point)):
            point = planning.trajectory_point[i]
            nextpoint = planning.trajectory_point[i + 1]
            point.ddkappa = (nextpoint.dkappa - point.dkappa) / (nextpoint.s - point.s)

        writer.write(planning)

        while not self.is_env_ready:
            time.sleep(0.1)  # second

        next_state = self.semantic_map()

        if self.arrival or self.violation_rule or self.collision:
            done = True
        else:
            done = False

        # Add debug information to info when in demand
        info = None

        return next_state, self.reward, done, info

    def reset(self):
        self.close()
        self.__init__()
        return self.state, self.hidden

    def close(self):
        # key input: q
        keyboard.press_and_release('q')
        pass

    def semantic_map():
        # TODO (Jinyun): generate semantic_map/img_feature
        return self.state

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
        self.is_env_ready = True

    def callback_chassis(self. entity):
        self.speed = entity.speed_mps
