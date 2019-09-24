#!/usr/bin/env python
# -*- coding: UTF-8-*-

from imu_speed import ImuSpeed


class ImuSpeedAcc:

    def __init__(self, is_lateral=False):
        self.timestamp_list = []
        self.acc_list = []
        self.imu_speed = ImuSpeed(is_lateral)

    def add(self, location_est):
        self.imu_speed.add(location_est)
        speed_timestamp_list = self.imu_speed.get_timestamp_list()

        index_50ms = len(speed_timestamp_list) - 1
        found_index_50ms = False
        last_timestamp = speed_timestamp_list[-1]
        while index_50ms >= 0:
            current_timestamp = speed_timestamp_list[index_50ms]
            if (last_timestamp - current_timestamp) >= 0.05:
                found_index_50ms = True
                break
            index_50ms -= 1

        if found_index_50ms:
            speed_list = self.imu_speed.get_speed_list()
            acc = (speed_list[-1] - speed_list[index_50ms]) / \
                  (speed_timestamp_list[-1] - speed_timestamp_list[index_50ms])
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
