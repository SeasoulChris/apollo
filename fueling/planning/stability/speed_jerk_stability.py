#!/usr/bin/env python
# -*- coding: UTF-8-*-

from fueling.planning.stability.imu_speed import ImuSpeed
from fueling.planning.stability.imu_speed_jerk import ImuSpeedJerk


class SpeedJerkStability:

    def __init__(self):
        self.speed_jerk_cnt = {}
        self.jerk_processor = ImuSpeedJerk()
        self.speed_processor = ImuSpeed()

    def add(self, location_est):
        self.speed_processor.add(location_est)
        self.jerk_processor.add(location_est)

    def get_speed_jerk_cnt(self):
        speed_list = self.speed_processor.get_speed_list()
        jerk_list = self.jerk_processor.get_jerk_list()

        grid_speed_list = self.grid(speed_list)
        grid_jerk_list = self.grid(jerk_list)

        grid_speed_list = grid_speed_list[-1 * len(grid_jerk_list):]

        for i in range(len(grid_speed_list)):
            speed = grid_speed_list[i]
            jerk = grid_jerk_list[i]

            if speed in self.speed_jerk_cnt:
                if jerk in self.speed_jerk_cnt[speed]:
                    self.speed_jerk_cnt[speed][jerk] += 1
                else:
                    self.speed_jerk_cnt[speed][jerk] = 1
            else:
                self.speed_jerk_cnt[speed] = {jerk: 1}

        return self.speed_jerk_cnt

    @staticmethod
    def grid(data_list):
        data_grid = []
        for data in data_list:
            data_grid.append(int(round(data)))
        return data_grid
