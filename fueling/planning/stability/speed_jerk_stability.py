#!/usr/bin/env python

###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

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
