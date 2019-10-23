#!/usr/bin/env python
from fueling.planning.stability.libs.planning_kappa import PlanningKappa
from fueling.planning.stability.libs.planning_speed import PlanningSpeed


class PlanningAv:

    def __init__(self):
        self.timestamp_list = []
        self.av_list = []

        self.planning_speed_processor = PlanningSpeed()
        self.planning_kappa_processor = PlanningKappa()
        self.last_speed = None
        self.last_timestamp = None

    def add(self, planning_pb):
        self.planning_speed_processor.add(planning_pb)
        self.planning_kappa_processor.add(planning_pb)

        timestamp_sec = planning_pb.header.timestamp_sec
        relative_time = planning_pb.debug.planning_data.init_point.relative_time
        current_timestamp = timestamp_sec + relative_time
        current_speed = self.planning_speed_processor.get_lastest_speed()
        current_kappa = self.planning_kappa_processor.get_lastest_kappa()

        av = current_speed * current_kappa

        self.timestamp_list.append(current_timestamp)
        self.av_list.append(av)

        self.last_timestamp = current_timestamp
        self.last_speed = current_speed

    def get_av_list(self):
        return self.av_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_av(self):
        if len(self.av_list) > 0:
            return self.av_list[-1]
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
            processor = PlanningAv()
            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/planning"]
            for data in reader.read(topics):
                if "planning" in data:
                    planning_pb = data["planning"]
                    processor.add(planning_pb)

            data_x = processor.get_timestamp_list()
            data_y = processor.get_av_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)
    plt.show()
