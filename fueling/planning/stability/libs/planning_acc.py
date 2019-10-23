#!/usr/bin/env python
from fueling.planning.stability.libs.planning_speed import PlanningSpeed


class PlanningAcc:

    def __init__(self):
        self.timestamp_list = []
        self.derived_acc_list = []
        self.origin_acc_list = []

        self.planning_speed_processor = PlanningSpeed()
        self.last_speed = None
        self.last_timestamp = None

    def add(self, planning_pb):
        self.planning_speed_processor.add(planning_pb)

        timestamp_sec = planning_pb.header.timestamp_sec
        relative_time = planning_pb.debug.planning_data.init_point.relative_time
        current_timestamp = timestamp_sec + relative_time
        current_speed = self.planning_speed_processor.get_lastest_speed()

        acc = planning_pb.debug.planning_data.init_point.a

        if self.last_timestamp is not None and self.last_speed is not None:
            delta_speed = current_speed - self.last_speed
            delta_t = current_timestamp - self.last_timestamp
            derived_acc = delta_speed / float(delta_t)
            self.derived_acc_list.append(derived_acc)
        else:
            self.derived_acc_list.append(acc)
        self.timestamp_list.append(current_timestamp)
        self.origin_acc_list.append(acc)

        self.last_timestamp = current_timestamp
        self.last_speed = current_speed

    def get_derived_acc_list(self):
        return self.derived_acc_list

    def get_origin_acc_list(self):
        return self.origin_acc_list

    def get_timestamp_list(self):
        return self.timestamp_list

    def get_lastest_origin_acc(self):
        if len(self.origin_acc_list) > 0:
            return self.origin_acc_list[-1]
        else:
            return None

    def get_lastest_derived_acc(self):
        if len(self.derived_acc_list) > 0:
            return self.derived_acc_list[-1]
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
            processor = PlanningAcc()
            last_pose_data = None
            last_chassis_data = None
            topics = ["/apollo/planning"]
            for data in reader.read(topics):
                if "planning" in data:
                    planning_pb = data["planning"]
                    processor.add(planning_pb)

            data_x = processor.get_timestamp_list()
            data_y = processor.get_origin_acc_list()
            ax.scatter(data_x, data_y, c=color, marker=marker, alpha=0.4)
            data_y2 = processor.get_derived_acc_list()
            ax.scatter(data_x, data_y2, c='b', marker="x", alpha=0.4)
    plt.show()
