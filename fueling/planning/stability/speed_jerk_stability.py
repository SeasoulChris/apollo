#!/usr/bin/env python
# -*- coding: UTF-8-*-

from collections import defaultdict

from fueling.planning.stability.imu_speed import ImuSpeed
from fueling.planning.stability.imu_speed_jerk import ImuSpeedJerk


class SpeedJerkStability(object):

    def __init__(self, is_lateral=False):
        self.speed_jerk_cnt = defaultdict(lambda: defaultdict(int))
        self.jerk_processor = ImuSpeedJerk(is_lateral)
        self.speed_processor = ImuSpeed(is_lateral)

    def add(self, location_est):
        self.speed_processor.add(location_est)
        self.jerk_processor.add(location_est)

    def get_speed_jerk_cnt(self):
        speed_list = self.speed_processor.get_speed_list()
        jerk_list = self.jerk_processor.get_jerk_list()
        # Return early when either list is empty.
        if not speed_list or not jerk_list:
            return {}

        grid_speed_list = self.grid(speed_list)
        grid_jerk_list = self.grid(jerk_list)
        grid_speed_list = grid_speed_list[-len(grid_jerk_list):]
        for i in range(len(grid_speed_list)):
            speed = grid_speed_list[i]
            jerk = grid_jerk_list[i]
            self.speed_jerk_cnt[speed][jerk] += 1
        return self.speed_jerk_cnt

    @staticmethod
    def grid(data_list):
        return [int(round(data)) for data in data_list]
