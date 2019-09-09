#!/usr/bin/env python
# -*- coding: UTF-8-*-

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
