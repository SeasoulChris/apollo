#!/usr/bin/env python

import sys

from modules.canbus.proto import chassis_pb2
from modules.localization.proto import localization_pb2
from modules.planning.proto import planning_pb2

import fueling.common.record_utils as record_utils


class RecordItemReader:
    def __init__(self, record_file):
        self.record_file = record_file

    def read(self, topics):
        reader = record_utils.read_record([
            record_utils.CHASSIS_CHANNEL,
            record_utils.LOCALIZATION_CHANNEL,
            record_utils.PLANNING_CHANNEL,
        ])
        for msg in reader(self.record_file):
            if msg.topic == record_utils.CHASSIS_CHANNEL:
                chassis = chassis_pb2.Chassis()
                chassis.ParseFromString(msg.message)
                data = {"chassis": chassis}
                yield data

            if msg.topic == record_utils.LOCALIZATION_CHANNEL:
                location_est = localization_pb2.LocalizationEstimate()
                location_est.ParseFromString(msg.message)
                data = {"pose": location_est}
                yield data

            if msg.topic == record_utils.PLANNING_CHANNEL:
                planning = planning_pb2.ADCTrajectory()
                planning.ParseFromString(msg.message)
                data = {"planning": planning}
                yield data
