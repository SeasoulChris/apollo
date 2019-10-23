#!/usr/bin/env python
from fueling.planning.stability.libs.planning_acc import PlanningAcc
from fueling.planning.stability.libs.planning_av import PlanningAv
from fueling.planning.stability.libs.planning_dav import PlanningDav
from fueling.planning.stability.libs.planning_speed import PlanningSpeed


class PlanningLatJerk:

    def __init__(self):
        self.timestamp_list = []
        self.lat_jerk_list = []

        self.planning_av_processor = PlanningAv()
        self.planning_dav_processor = PlanningDav()
        self.planning_speed_processor = PlanningSpeed()
        self.planning_acc_processor = PlanningAcc()

    def add(self, planning_pb):
        self.planning_av_processor.add(planning_pb)
        self.planning_dav_processor.add(planning_pb)
        self.planning_speed_processor.add(planning_pb)
        self.planning_acc_processor.add(planning_pb)

        timestamp_sec = planning_pb.header.timestamp_sec
        relative_time = planning_pb.debug.planning_data.init_point.relative_time
        current_timestamp = timestamp_sec + relative_time

        av = self.planning_av_processor.get_lastest_av()
        dav = self.planning_dav_processor.get_lastest_dav()
        speed = self.planning_speed_processor.get_lastest_speed()
        acc = self.planning_acc_processor.get_lastest_derived_acc()

        if av is not None and dav is not None and speed is not None and acc is not None:
            lat_jerk = -1 * (dav * speed + av * acc)
            self.timestamp_list.append(current_timestamp)
            self.lat_jerk_list.append(lat_jerk)

    def get_lat_jerk_list(self):
        return self.lat_jerk_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_lat_jerk(self):
        if len(self.lat_jerk_list) > 0:
            return self.lat_jerk_list[-1]
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
        for fn in fns:
            reader = RecordItemReader(folder + "/" + fn)
            processor = PlanningLatJerk()
            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/planning"]
            for data in reader.read(topics):
                if "planning" in data:
                    planning_pb = data["planning"]
                    processor.add(planning_pb)

            data_x = processor.get_timestamp_list()
            data_y = processor.get_lat_jerk_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)

    plt.show()
