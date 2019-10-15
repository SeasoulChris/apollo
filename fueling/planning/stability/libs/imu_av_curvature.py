#!/usr/bin/env python

from fueling.planning.stability.libs.imu_angular_velocity import ImuAngularVelocity
from fueling.planning.stability.libs.imu_speed import ImuSpeed


class ImuAvCurvature:
    def __init__(self):
        self.timestamp_list = []
        self.curvature_list = []

        self.last_angular_velocity_z = None

        self.imu_angular_velocity = ImuAngularVelocity()
        self.imu_speed = ImuSpeed()

    def add(self, location_est):
        timestamp_sec = location_est.header.timestamp_sec

        self.imu_angular_velocity.add(location_est)
        self.imu_speed.add(location_est)

        angular_velocity_z \
            = self.imu_angular_velocity.get_latest_corrected_angular_velocity()
        speed_mps = self.imu_speed.get_lastest_speed()
        if speed_mps > 0.03:
            kappa = angular_velocity_z / speed_mps
        else:
            kappa = 0

        self.timestamp_list.append(timestamp_sec)
        self.curvature_list.append(kappa)

        self.last_angular_velocity_z = angular_velocity_z

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_curvature_list(self):
        return self.curvature_list

    def get_last_timestamp(self):
        if len(self.timestamp_list) > 0:
            return self.timestamp_list[-1]
        return None

    def get_last_curvature(self):
        if len(self.curvature_list) > 0:
            return self.curvature_list[-1]
        return None


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from os import listdir
    from os.path import isfile, join
    from fueling.planning.stability.libs.record_reader import RecordItemReader

    folders = sys.argv[1:]
    fig, ax = plt.subplots()
    colors = ["g", "b", "r", "m", "y"]
    markers = ["o", "o", "o", "o"]
    for i in range(len(folders)):
        folder = folders[i]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        fns = [f for f in listdir(folder) if isfile(join(folder, f))]
        fns.sort()
        for fn in fns:
            print(fn)
            reader = RecordItemReader(folder + "/" + fn)
            curvature_processor = ImuAvCurvature()
            speed_processor = ImuSpeed()
            av_processor = ImuAngularVelocity()
            last_pose_data = None
            last_chassis_data = None
            for data in reader.read(["/apollo/localization/pose"]):
                if "pose" in data:
                    last_pose_data = data["pose"]
                    curvature_processor.add(last_pose_data)
                    speed_processor.add(last_pose_data)
                    av_processor.add(last_pose_data)

            data_x = curvature_processor.get_timestamp_list()
            data_y = curvature_processor.get_curvature_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)

            data_x = speed_processor.get_timestamp_list()
            data_y = speed_processor.get_speed_list()
            ax.scatter(data_x, data_y, c='r', marker=marker, alpha=0.4)

            data_x = av_processor.get_timestamp_list()
            data_y = av_processor.get_corrected_anglular_velocity_list()
            ax.scatter(data_x, data_y, c='b', marker=marker, alpha=0.4)

    plt.show()
