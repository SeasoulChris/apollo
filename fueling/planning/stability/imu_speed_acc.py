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

from imu_speed import ImuSpeed


class ImuSpeedAcc:

    def __init__(self):
        self.timestamp_list = []
        self.acc_list = []
        self.imu_speed = ImuSpeed()

    def add(self, location_est):
        self.imu_speed.add(location_est)
        speed_timestamp_list = self.imu_speed.get_timestamp_list()
        if len(speed_timestamp_list) > 5:
            speed_list = self.imu_speed.get_speed_list()
            acc = (speed_list[-1] - speed_list[-5]) / \
                  (speed_timestamp_list[-1] - speed_timestamp_list[-5])
            self.acc_list.append(acc)
            self.timestamp_list.append(speed_timestamp_list[-1])

    def get_acc_list(self):
        return self.acc_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_acc(self):
        if len(self.acc_list) > 0:
            return self.acc_list[-1]
        else:
            return None

    def get_lastest_timestamp(self):
        if len(self.timestamp_list) > 0:
            return self.timestamp_list[-1]
        else:
            return None
