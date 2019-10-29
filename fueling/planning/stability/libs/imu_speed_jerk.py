#!/usr/bin/env python

from fueling.planning.stability.libs.imu_speed_acc import ImuSpeedAcc


class ImuSpeedJerk:

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


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    from fueling.planning.stability.libs.record_reader import RecordItemReader

    folders = sys.argv[1:]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    colors = ["g", "b", "r", "m", "y"]
    markers = ["o", "o", "o", "o"]
    for i in range(len(folders)):
        x = []
        y = []
        folder = folders[i]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        fns = sorted([f for f in listdir(folder) if isfile(join(folder, f))])
        for fn in fns:
            reader = RecordItemReader(folder + "/" + fn)
            lat_processor = ImuSpeedJerk(True)
            lon_processor = ImuSpeedJerk(False)

            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/localization/pose"]
            for data in reader.read(topics):
                if "pose" in data:
                    last_pose_data = data["pose"]
                    lat_processor.add(last_pose_data)
                    lon_processor.add(last_pose_data)

            data_x = lat_processor.get_timestamp_list()
            data_y = lat_processor.get_jerk_list()
            ax1.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)

            data_x1 = lon_processor.get_timestamp_list()
            data_y1 = lon_processor.get_jerk_list()
            ax2.scatter(data_x1, data_y1, c=color, marker=marker, alpha=0.4)

            x.extend(data_x)
            y.extend(data_y)

        if len(x) <= 0:
            continue

        # ax.plot(x, y, c=color, alpha=0.4)

        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Lat Jerk')

    plt.show()
