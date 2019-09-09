#!/usr/bin/env python
# -*- coding: UTF-8-*-

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
