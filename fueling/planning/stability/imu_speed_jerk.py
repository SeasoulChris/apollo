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

from imu_speed_acc import ImuSpeedAcc


class ImuSpeedJerk:

    def __init__(self):
        self.timestamp_list = []
        self.jerk_list = []
        self.imu_speed_acc = ImuSpeedAcc()

    def add(self, location_est):
        self.imu_speed_acc.add(location_est)
        acc_timestamp_list = self.imu_speed_acc.get_timestamp_list()
        if len(acc_timestamp_list) > 50:
            acc_list = self.imu_speed_acc.get_acc_list()
            jerk = (acc_list[-1] - acc_list[-50]) / \
                   (acc_timestamp_list[-1] - acc_timestamp_list[-50])
            self.jerk_list.append(jerk)
            self.timestamp_list.append(acc_timestamp_list[-1])

    def get_jerk_list(self):
        return self.jerk_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_jerk(self):
        if len(self.jerk_list) > 0:
            return self.jerk_list[-1]
        else:
            return None

    def get_lastest_timestamp(self):
        if len(self.timestamp_list) > 0:
            return self.timestamp_list[-1]
        else:
            return None
