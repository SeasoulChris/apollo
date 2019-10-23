#!/usr/bin/env python
from fueling.planning.stability.libs.planning_acc import PlanningAcc


class PlanningJerk:

    def __init__(self):
        self.timestamp_list = []
        self.derived_jerk_list = []
        self.origin_jerk_list = []

        self.planning_acc_processor = PlanningAcc()
        self.last_acc = None
        self.last_timestamp = None

    def add(self, planning_pb):
        self.planning_acc_processor.add(planning_pb)
        timestamp_sec = planning_pb.header.timestamp_sec
        relative_time = planning_pb.debug.planning_data.init_point.relative_time
        current_timestamp = timestamp_sec + relative_time
        current_acc = self.planning_acc_processor.get_lastest_derived_acc()

        jerk = planning_pb.debug.planning_data.init_point.da

        if self.last_timestamp is not None and self.last_acc is not None:
            delta_acc = current_acc - self.last_acc
            delta_t = current_timestamp - self.last_timestamp
            derived_jerk = delta_acc / float(delta_t)
            self.derived_jerk_list.append(derived_jerk)
        else:
            self.derived_jerk_list.append(jerk)
        self.timestamp_list.append(current_timestamp)
        self.origin_jerk_list.append(jerk)

        self.last_timestamp = current_timestamp
        self.last_acc = current_acc

    def get_derived_jerk_list(self):
        return self.derived_jerk_list

    def get_origin_jerk_list(self):
        return self.origin_jerk_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_origin_jerk(self):
        if len(self.origin_jerk_list) > 0:
            return self.origin_jerk_list[-1]
        else:
            return None

    def get_lastest_derived_jerk(self):
        if len(self.derived_jerk_list) > 0:
            return self.derived_jerk_list[-1]
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
            processor = PlanningJerk()
            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/planning"]
            for data in reader.read(topics):
                if "planning" in data:
                    planning_pb = data["planning"]
                    processor.add(planning_pb)

            data_x = processor.get_timestamp_list()
            data_y = processor.get_origin_jerk_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)
            data_y2 = processor.get_derived_jerk_list()
            ax.scatter(data_x, data_y2, c='b', marker="x", alpha=0.4)
    plt.show()
