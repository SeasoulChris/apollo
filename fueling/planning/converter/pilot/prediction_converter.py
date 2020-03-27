#!/usr/bin/env python

from py_proto.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle
from py_proto.modules.prediction.proto.prediction_obstacle_pb2 import PredictionObstacles


class PredictionConverter:
    def __init__(self):
        pass

    def convert(self, polit_pediction):
        apollo_prediction = PredictionObstacles()

        apollo_prediction.header.ParseFromString(polit_pediction.header.SerializeToString())

        for pilot_prediction_obs in polit_pediction.prediction_obstacle:
            apollo_pred_obs = apollo_prediction.prediction_obstacle.add()
            # 1
            apollo_pred_obs.perception_obstacle.CopyFrom(
                self.perception_obstacle_convert(pilot_prediction_obs.perception_obstacle))
            # 2
            apollo_pred_obs.timestamp = pilot_prediction_obs.time_stamp

            # 3
            apollo_pred_obs.predicted_period = pilot_prediction_obs.predicted_period

            # 4
            for pilot_traj in pilot_prediction_obs.trajectory:
                apollo_traj = apollo_pred_obs.trajectory.add()
                apollo_traj.ParseFromString(pilot_traj.SerializeToString())

            # 5

        return apollo_prediction

    def perception_obstacle_convert(self, pilot_perception_obstacle):
        apollo_perception_obs = PerceptionObstacle()
        # 1
        apollo_perception_obs.id = pilot_perception_obstacle.id
        # 2
        apollo_perception_obs.position.x = pilot_perception_obstacle.position.x
        apollo_perception_obs.position.y = pilot_perception_obstacle.position.y
        apollo_perception_obs.position.z = pilot_perception_obstacle.position.z
        # 3
        apollo_perception_obs.theta = pilot_perception_obstacle.theta
        # 4
        apollo_perception_obs.velocity.x = pilot_perception_obstacle.velocity.x
        apollo_perception_obs.velocity.x = pilot_perception_obstacle.velocity.x
        apollo_perception_obs.velocity.y = pilot_perception_obstacle.velocity.y
        # 5,6,7
        apollo_perception_obs.length = pilot_perception_obstacle.length
        apollo_perception_obs.width = pilot_perception_obstacle.width
        apollo_perception_obs.height = pilot_perception_obstacle.height
        # 8
        for point in pilot_perception_obstacle.polygon_point:
            apollo_point = apollo_perception_obs.polygon_point.add()
            apollo_point.x = point.x
            apollo_point.y = point.y
            apollo_point.z = point.z
        # 9
        apollo_perception_obs.tracking_time = pilot_perception_obstacle.tracking_time
        # 10
        apollo_perception_obs.type = pilot_perception_obstacle.type
        # 11
        apollo_perception_obs.timestamp = pilot_perception_obstacle.timestamp
        # 12
        # 13
        apollo_perception_obs.anchor_point.x = pilot_perception_obstacle.anchor_point.x
        apollo_perception_obs.anchor_point.y = pilot_perception_obstacle.anchor_point.y
        apollo_perception_obs.anchor_point.z = pilot_perception_obstacle.anchor_point.z
        # 16
        apollo_perception_obs.height_above_ground = pilot_perception_obstacle.height_above_ground

        return apollo_perception_obs
