#!/usr/bin/env python
from fueling.planning.stability.libs.planning_av import PlanningAv


class PlanningDav:

    def __init__(self):
        self.timestamp_list = []
        self.dav_list = []

        self.planning_av_processor = PlanningAv()
        self.last_av = None
        self.last_timestamp = None

    def add(self, planning_pb):
        self.planning_av_processor.add(planning_pb)

        timestamp_sec = planning_pb.header.timestamp_sec
        relative_time = planning_pb.debug.planning_data.init_point.relative_time
        current_timestamp = timestamp_sec + relative_time
        current_av = self.planning_av_processor.get_lastest_av()

        acc = planning_pb.debug.planning_data.init_point.a

        if self.last_timestamp is not None and self.last_av is not None:
            delta_av = current_av - self.last_av
            delta_t = current_timestamp - self.last_timestamp
            dav = 0 if delta_t <= 0 else delta_av / float(delta_t)
            self.dav_list.append(dav)
            self.timestamp_list.append(current_timestamp)

        self.last_timestamp = current_timestamp
        self.last_av = current_av

    def get_dav_list(self):
        return self.dav_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_dav(self):
        if len(self.dav_list) > 0:
            return self.dav_list[-1]
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
            processor = PlanningDav()
            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/planning"]
            for data in reader.read(topics):
                if "planning" in data:
                    planning_pb = data["planning"]
                    processor.add(planning_pb)

            data_x = processor.get_timestamp_list()
            data_y = processor.get_dav_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)
    plt.show()
