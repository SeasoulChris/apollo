#!/usr/bin/env python

from modules.prediction.proto import prediction_obstacle_pb2


class PredictionAnalyzer:
    def __init__(self):
        self.prediction = None

    def update(self, prediction_msg):
        self.prediction = prediction_obstacle_pb2.PredictionObstacles()
        self.prediction.ParseFromString(prediction_msg.message)

    def get_last_prediction_timestamp(self):
        if self.prediction is None:
            return 0
        return self.prediction.header.timestamp_sec
