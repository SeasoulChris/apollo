#!/usr/bin/env python

from modules.perception.proto import perception_obstacle_pb2


class PerceptionAnalyzer:
    def __init__(self):
        self.perception_obstacle = None

    def update(self, perception_obstacle_msg):
        self.perception_obstacle = perception_obstacle_pb2.PerceptionObstacles()
        self.perception_obstacle.ParseFromString(perception_obstacle_msg.message)

    def get_last_perception_timestamp(self):
        if self.perception_obstacle is None:
            return 0
        return self.perception_obstacle.header.timestamp_sec

    @staticmethod
    def get_msg_timstamp(msg):
        perception = perception_obstacle_pb2.PerceptionObstacles()
        perception.ParseFromString(msg.message)
        return perception.header.timestamp_sec
