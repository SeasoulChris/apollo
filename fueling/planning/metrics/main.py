#!/usr/bin/env python
import argparse
import json
import sys

import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    from cyber_py.record_py3 import RecordReader
else:
    from cyber_py.record import RecordReader
from modules.control.proto import control_cmd_pb2
from modules.planning.proto import planning_pb2
from modules.canbus.proto import chassis_pb2
from modules.drivers.proto import pointcloud_pb2
from module_control_analyzer import ControlAnalyzer
from module_planning_analyzer import PlannigAnalyzer
from modules.perception.proto import perception_obstacle_pb2
from modules.prediction.proto import prediction_obstacle_pb2
from fueling.planning.metrics.lidar_endtoend_analyzer import LidarEndToEndAnalyzer
from fueling.planning.stability.grading.planning_stability_grader import Grader


def process(control_analyzer, planning_analyzer, lidar_endtoend_analyzer,
            is_simulation, plot_planning_path, plot_planning_refpath, all_data):
    is_auto_drive = False

    for msg in reader.read_messages():
        if msg.topic == "/apollo/canbus/chassis":
            chassis = chassis_pb2.Chassis()
            chassis.ParseFromString(msg.message)
            if chassis.driving_mode == \
                    chassis_pb2.Chassis.COMPLETE_AUTO_DRIVE:
                is_auto_drive = True
            else:
                is_auto_drive = False

        if msg.topic == "/apollo/control":
            if (not is_auto_drive and not all_data) or \
                    is_simulation or plot_planning_path or plot_planning_refpath:
                continue
            control_cmd = control_cmd_pb2.ControlCommand()
            control_cmd.ParseFromString(msg.message)
            control_analyzer.put(control_cmd)
            lidar_endtoend_analyzer.put_pb('control', control_cmd)

        if msg.topic == "/apollo/planning":
            if (not is_auto_drive) and (not all_data):
                continue
            adc_trajectory = planning_pb2.ADCTrajectory()
            adc_trajectory.ParseFromString(msg.message)
            planning_analyzer.put(adc_trajectory)
            lidar_endtoend_analyzer.put_pb('planning', adc_trajectory)

            if plot_planning_path:
                planning_analyzer.plot_path(plt, adc_trajectory)
            if plot_planning_refpath:
                planning_analyzer.plot_refpath(plt, adc_trajectory)

        if msg.topic == "/apollo/sensor/velodyne64/compensator/PointCloud2" or \
                msg.topic == "/apollo/sensor/lidar128/compensator/PointCloud2":
            if ((not is_auto_drive) and (not all_data)) or is_simulation or \
                    plot_planning_path or plot_planning_refpath:
                continue
            point_cloud = pointcloud_pb2.PointCloud()
            point_cloud.ParseFromString(msg.message)
            lidar_endtoend_analyzer.put_lidar(point_cloud)

        if msg.topic == "/apollo/perception/obstacles":
            if ((not is_auto_drive) and (not all_data)) or is_simulation or \
                    plot_planning_path or plot_planning_refpath:
                continue
            perception = perception_obstacle_pb2.PerceptionObstacles()
            perception.ParseFromString(msg.message)
            lidar_endtoend_analyzer.put_pb('perception', perception)

        if msg.topic == "/apollo/prediction":
            if ((not is_auto_drive) and (not all_data)) or is_simulation or \
                    plot_planning_path or plot_planning_refpath:
                continue
            prediction = prediction_obstacle_pb2.PredictionObstacles()
            prediction.ParseFromString(msg.message)
            lidar_endtoend_analyzer.put_pb('prediction', prediction)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py record_file")

    parser = argparse.ArgumentParser(
        description="Recode Analyzer is a tool to analyze record files.",
        prog="main.py")

    parser.add_argument(
        "-f", "--file", action="store", type=str, required=True,
        help="Specify the record file for analysis.")

    parser.add_argument(
        "-sim", "--simulation", action="store_const", const=True,
        help="For dreamland API call")

    parser.add_argument(
        "-path", "--planningpath", action="store_const", const=True,
        help="plot planing paths in cartesian coordinate.")

    parser.add_argument(
        "-refpath", "--planningrefpath", action="store_const", const=True,
        help="plot planing reference paths in cartesian coordinate.")

    parser.add_argument(
        "-a", "--alldata", action="store_const", const=True,
        help="Analyze all data (both auto and manual), otherwise auto data only without this option.")

    parser.add_argument(
        "-acc", "--showacc", action="store_const", const=True,
        help="Analyze all data (both auto and manual), otherwise auto data only without this option.")

    args = parser.parse_args()

    record_file = args.file
    reader = RecordReader(record_file)

    control_analyzer = ControlAnalyzer()
    planning_analyzer = PlannigAnalyzer(args)
    lidar_endtoend_analyzer = LidarEndToEndAnalyzer()

    process(control_analyzer, planning_analyzer,
            lidar_endtoend_analyzer, args.simulation, args.planningpath,
            args.planningrefpath, args.alldata)

    if args.simulation:
        score_list = Grader().grade_record_file(record_file)

        results = planning_analyzer.get_sim_results()
        if len(results) > 0:
            stability_score = dict()
            stability_score["avg"] = sum(score_list) / float(len(score_list))
            stability_score["max"] = max(score_list)
            stability_score["min"] = min(score_list)
            results["stability"] = stability_score
        print(json.dumps(results))
    elif args.planningpath or args.planningrefpath:
        plt.axis('equal')
        plt.show()
    else:
        control_analyzer.print_latency_statistics()
        planning_analyzer.print_latency_statistics()
        lidar_endtoend_analyzer.print_endtoend_latency()
