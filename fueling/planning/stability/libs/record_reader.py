#!/usr/bin/env python

import sys

if sys.version_info[0] >= 3:
    from cyber_py3.record import RecordReader
else:
    from cyber_py.record import RecordReader

from modules.canbus.proto import chassis_pb2
from modules.localization.proto import localization_pb2
from modules.planning.proto import planning_pb2


class RecordItemReader:
    def __init__(self, record_file):
        self.record_file = record_file

    def read(self, topics):
        reader = RecordReader(self.record_file)
        for msg in reader.read_messages():
            if msg.topic not in topics:
                continue
            if msg.topic == "/apollo/canbus/chassis":
                chassis = chassis_pb2.Chassis()
                chassis.ParseFromString(msg.message)
                data = {"chassis": chassis}
                yield data

            if msg.topic == "/apollo/localization/pose":
                location_est = localization_pb2.LocalizationEstimate()
                location_est.ParseFromString(msg.message)
                data = {"pose": location_est}
                yield data

            if msg.topic == "/apollo/planning":
                planning = planning_pb2.ADCTrajectory()
                planning.ParseFromString(msg.message)
                data = {"planning": planning}
                yield data
