#!/usr/bin/env python
"""Localization Analyzer."""

from modules.localization.proto import localization_pb2
from modules.perception.proto import perception_obstacle_pb2
from modules.canbus.proto import chassis_pb2
from modules.prediction.proto import prediction_obstacle_pb2


class MsgFreqAnalyzer:
    def __init__(self):

        self.localization_timestamp = -1
        self.perception_timestamp = -1
        self.prediction_timestamp = -1
        self.chassis_timestamp = -1
        self.is_debug = True
        self.msgs_set = []

    def process(self, msgs):
        self.msgs_set = []
        msgs_temp = []
        perception_msg_cnt = 0

        for msg in msgs:
            if msg.topic == '/apollo/perception/obstacles':
                perception_obstacle_pb = perception_obstacle_pb2.PerceptionObstacles()
                perception_obstacle_pb.ParseFromString(msg.message)
                timestamp = perception_obstacle_pb.header.timestamp_sec

                if abs(timestamp - self.chassis_timestamp) > 0.03 \
                        or abs(timestamp - self.chassis_timestamp) > 0.03 \
                        or abs(timestamp - self.prediction_timestamp) > 0.3 \
                        or abs(timestamp - self.perception_timestamp) > 0.3:
                    if self.is_debug:
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        print(timestamp - self.chassis_timestamp)
                        print(timestamp - self.prediction_timestamp)
                        print(timestamp - self.prediction_timestamp)
                        print(timestamp - self.perception_timestamp)
                        print("perception_msg_cnt = " + str(perception_msg_cnt))
                        print("-------------")
                    if perception_msg_cnt > 80:
                        self.msgs_set.append(msgs_temp)
                    msgs_temp = []
                    perception_msg_cnt = 0

                self.perception_timestamp = timestamp
                perception_msg_cnt += 1

            if msg.topic == '/apollo/canbus/chassis':
                chassis = chassis_pb2.Chassis()
                chassis.ParseFromString(msg.message)
                self.chassis_timestamp = chassis.header.timestamp_sec

            if msg.topic == '/apollo/localization/pose':
                localization_estimate = localization_pb2.LocalizationEstimate()
                localization_estimate.ParseFromString(msg.message)
                timestamp = localization_estimate.header.timestamp_sec
                self.localization_timestamp = timestamp

            if msg.topic == '/apollo/prediction':
                prediction = prediction_obstacle_pb2.PredictionObstacles()
                prediction.ParseFromString(msg.message)
                timestamp = prediction.header.timestamp_sec
                self.prediction_timestamp = timestamp

            if msg.topic == '/apollo/routing_response_history':
                pass

            if msg.topic == '/apollo/routing_response':
                pass

            if msg.topic == '/apollo/perception/traffic_light':
                pass
            msgs_temp.append(msg)

        if perception_msg_cnt > 80:
            self.msgs_set.append(msgs_temp)
        if self.is_debug:
            print([len(msgs) for msgs in self.msgs_set])
        return self.msgs_set


if __name__ == "__main__":
    import sys
    from cyber_py3.record import RecordReader

    fn = sys.argv[1]
    reader = RecordReader(fn)
    msgs = [msg for msg in reader.read_messages()]
    analyzer = MsgFreqAnalyzer()
    analyzer.process(msgs)
