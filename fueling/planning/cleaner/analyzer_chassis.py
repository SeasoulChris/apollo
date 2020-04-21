#!/usr/bin/env python

from modules.canbus.proto import chassis_pb2


class ChassisAnalyzer:
    def __init__(self):
        self.chassis = None

    def update(self, chassis_msg):
        self.chassis = chassis_pb2.Chassis()
        self.chassis.ParseFromString(chassis_msg.message)

    def get_last_chassis_timestamp(self):
        if self.chassis is None:
            return 0
        return self.chassis.header.timestamp_sec
