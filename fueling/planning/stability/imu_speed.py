#!/usr/bin/env python
# -*- coding: UTF-8-*-

import math


class ImuSpeed:

    def __init__(self):
        self.timestamp_list = []
        self.speed_list = []

        self.last_speed_mps = None
        self.last_imu_speed = None

    def add(self, location_est):
        timestamp_sec = location_est.measurement_time
        self.timestamp_list.append(timestamp_sec)

        velocity = location_est.pose.linear_velocity
        heading = location_est.pose.heading
        speed = velocity.x * math.cos(heading) + velocity.y * math.sin(heading)
        self.speed_list.append(speed)

    def get_speed_list(self):
        return self.speed_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_speed(self):
        return self.speed_list[-1] if self.speed_list else None

    def get_lastest_timestamp(self):
        return self.timestamp_list[-1] if self.timestamp_list else None
