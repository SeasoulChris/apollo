import threading

import keyboard

from cyber.python.cyber_py3 import cyber, cyber_time


class ADSEnv(object):
    def __init__(self, hidden_size=128):
        self.hidden = generate_lstm_states()
        self.state = self.semantic_map()
        self.reward = 0

        # for callback_planning
        self.pointx = []
        self.pointy = []
        self.pointspeed = []
        self.pointtime = []
        self.pointtheta = []
        self.pointcurvature = []
        self.pointacceleration = []
        self.planningavailable = False
        self.lock = threading.Lock()

        cyber.init()
        rl_node = cyber.Node("rl_node")
        # TODO (Una/Yifei): Confirm the location of the proto
        gradingsub = rl_node.create_reader("/apollo/simulator",
                                           grading_result.FrameResult, self.callback_grading)
        planningsub = rl_node.create_reader("/apollo/planning",
                                            grading_result.FrameResult, self.callback_planning)

    def step(self, action):
        """return next state, reward, done, info"""
        # key input: space
        keyboard.press_and_release('space')
        # send planning msg (action)
        writer = rl_node.crete_writter("/apollo/planning", ADCTrajectory)
        planning = ADCTrajectory()
        planning.header.timestamp_sec = cyber_time.Time.now().to_sec()
        planning.header.module_name = "planning"
        planning.total_path_time = 2
        planning.trajectory_point = action
        planning.path_point = 10
        writer.write(planning)

        while not self.is_env_ready():
            time.sleep(0.1)  # second

        next_state = self.semantic_map()
        # TODO (Songyang): generate reward
        return next_state, reward, done, info

    def received(self, msg):
        """env msg (perception, prediction, chassis, ...)"""
        pass

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

    def callback_planning(self, entity):
        """
        New Planning Trajectory
        """
        basetime = entity.header.timestamp_sec
        numpoints = len(entity.trajectory_point)
        with self.lock:
            self.pointx = numpy.zeros(numpoints)
            self.pointy = numpy.zeros(numpoints)
            self.pointspeed = numpy.zeros(numpoints)
            self.pointtime = numpy.zeros(numpoints)
            self.pointtheta = numpy.zeros(numpoints)
            self.pointcurvature = numpy.zeros(numpoints)
            self.pointacceleration = numpy.zeros(numpoints)

            for idx in range(numpoints):
                self.pointx[idx] = entity.trajectory_point[idx].path_point.x
                self.pointy[idx] = entity.trajectory_point[idx].path_point.y
                self.pointspeed[idx] = entity.trajectory_point[idx].v
                self.pointtheta[idx] = entity.trajectory_point[
                    idx].path_point.theta
                self.pointcurvature[idx] = entity.trajectory_point[
                    idx].path_point.kappa
                self.pointacceleration[idx] = entity.trajectory_point[
                    idx].a
                self.pointtime[
                    idx] = entity.trajectory_point[idx].relative_time + basetime

        if numpoints == 0:
            self.planningavailable = False
        else:
            self.planningavailable = True

    def callback_grading(self, entity):
        self.reward = 0
        violation_rule = False
        for result in entity.detailed_result:
            if result.name == "Collision" and result.is_pass is False:
                self.reward -= 500
            if result.name == "DistanceToLaneCenter":
                dist_lane_center = result.score

            # test whether the vehicle violates the traffic rule
            if result.name == "AccelerationLimit" and result.is_pass is False:
                violation_rule = True
            if result.name == "RunRedLight" and result.is_pass is False:
                violation_rule = True
            if result.name == "SpeedLimit" and result.is_pass is False:
                violation_rule = True
            if result.name == "ChangeLaneAtJunction" and result.is_pass is False:
                violation_rule = True
            if result.name == "CrosswalkYieldToPedestrians" and result.is_pass is False:
                violation_rule = True
            if result.name == "RunStopSign" and result.is_pass is False:
                violation_rule = True

            if result.name == "DistanceToEnd":
                dist_end = result.score

        # TODO: confirm WIDTH_LANE
        if dist_lane_center >= 0.75 * WIDTH_LANE:
            self.reward -= 250
        if violation_rule:
            self.reward -= 250
        # TODO: check this threshold
        if dist_end <= 10:
            self.reward += 100
        self.reward -= 0.1 * dist_lane_center
