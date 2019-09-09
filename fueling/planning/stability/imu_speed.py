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

        speed = location_est.pose.linear_velocity.x \
                * math.cos(location_est.pose.heading) + \
                location_est.pose.linear_velocity.y * \
                math.sin(location_est.pose.heading)
        self.speed_list.append(speed)

    def get_speed_list(self):
        return self.speed_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_speed(self):
        if len(self.speed_list) > 0:
            return self.speed_list[-1]
        else:
            return None

    def get_lastest_timestamp(self):
        if len(self.timestamp_list) > 0:
            return self.timestamp_list[-1]
        else:
            return None
