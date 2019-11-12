#!/usr/bin/env python

from fueling.planning.metrics.common.statistical_analyzer import StatisticalAnalyzer
from fueling.planning.metrics.common.statistical_analyzer import PrintColors


class LidarEndToEndAnalyzer(object):
    """
    Control analyzer
    """

    def __init__(self):
        """
        Init
        """
        self.modules = ['control', 'planning', 'prediction', 'perception']
        self.endtoend_latency = {}
        self.unprocessed_lidar_timestamps = {}
        for m in self.modules:
            self.endtoend_latency[m] = []
            self.unprocessed_lidar_timestamps[m] = []

    def put_pb(self, module_name, pb_msg):
        """
        Put  data
        """
        if module_name not in self.unprocessed_lidar_timestamps:
            print(module_name, " is not supported")
            return

        if pb_msg.header.lidar_timestamp in \
                self.unprocessed_lidar_timestamps[module_name]:
            ind = self.unprocessed_lidar_timestamps[module_name].index(
                pb_msg.header.lidar_timestamp)
            del (self.unprocessed_lidar_timestamps[module_name][ind])
            self.endtoend_latency[module_name].append(
                (pb_msg.header.timestamp_sec -
                 pb_msg.header.lidar_timestamp * 1.0e-9) * 1000.0)

    def put_lidar(self, point_cloud):
        """
        Put lidar data
        """
        for m in self.modules:
            self.unprocessed_lidar_timestamps[m].append(
                point_cloud.header.lidar_timestamp)

    def print_endtoend_latency(self):
        """
        Print end to end latency
        """
        print("\n\n")
        for m in self.modules:
            print(PrintColors.HEADER + "* End to End (" + m
                  + ") Latency (ms)" + PrintColors.ENDC)
            analyzer = StatisticalAnalyzer()
            analyzer.print_statistical_results(self.endtoend_latency[m])

            print(PrintColors.FAIL + "  - MISS # OF LIDAR: " +
                  str(len(self.unprocessed_lidar_timestamps[m])) +
                  PrintColors.ENDC)
