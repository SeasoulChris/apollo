#!/usr/bin/env python
# -*- coding: UTF-8-*-

from fueling.planning.stability.imu_speed_acc import ImuSpeedAcc


class ImuSpeedJerk(object):

    def __init__(self, is_lateral=False):
        self.timestamp_list = []
        self.jerk_list = []
        self.imu_speed_acc = ImuSpeedAcc(is_lateral)

    def add(self, location_est):
        self.imu_speed_acc.add(location_est)
        acc_timestamp_list = self.imu_speed_acc.get_timestamp_list()
        if len(acc_timestamp_list) <= 0:
            return

        index_500ms = len(acc_timestamp_list) - 1
        found_index_500ms = False
        last_timestamp = acc_timestamp_list[-1]
        while index_500ms >= 0:
            current_timestamp = acc_timestamp_list[index_500ms]
            if (last_timestamp - current_timestamp) >= 0.5:
                found_index_500ms = True
                break
            index_500ms -= 1

        if found_index_500ms:
            acc_list = self.imu_speed_acc.get_acc_list()
            jerk = (acc_list[-1] - acc_list[index_500ms]) / \
                   (acc_timestamp_list[-1] - acc_timestamp_list[index_500ms])
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
