#!/usr/bin/env python3

import datetime
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from cyber_py3.record import RecordReader
from modules.localization.proto import localization_pb2
from modules.perception.proto import perception_obstacle_pb2
from modules.routing.proto.routing_pb2 import RoutingResponse

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from fueling.common.base_pipeline import BasePipeline

from fueling.planning.data_visualizer import mkz_plotter
from fueling.planning.data_visualizer import obstacle_plotter
from fueling.planning.data_visualizer import routing_plotter


class VisualPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        task = "2019-11-22-20-39-47/"
        self.dst_prefix = 'modules/planning/visual_data/' + task
        self.src_prefixs = [
            'modules/planning/cleaned_data/' + task
        ]

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        self.dst_prefix = '/fuel/data/planning/visual_data/'

        records = ['/fuel/testdata/data/small.record', '/fuel/testdata/data/small.record']
        for record in records:
            self.process_record(record)

    def run(self):
        """Run prod."""
        prefix = "/mnt/bos/"
        for task_folder in self.src_prefixs:
            task_folder = prefix + task_folder
            for filename in os.listdir(task_folder):
                file_path = os.path.join(task_folder, filename)
                if not os.path.isdir(file_path):
                    if record_utils.is_record_file(file_path):
                        logging.info(file_path)
                        self.process_record(file_path)
        return

        # RDD(record_path)
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix)).filter(record_utils.is_record_file)
            for prefix in self.src_prefixs])

        processed_records = records_rdd.map(self.process_record)
        logging.info('Processed {} records'.format(processed_records.count()))

    def process_record(self, src_record_fn):
        dst_img_fn = src_record_fn.replace("cleaned_data", "visual_data")
        dst_img_fn += ".jpg"
        logging.info(dst_img_fn)

        self.visualize(src_record_fn, dst_img_fn)

        return dst_img_fn

    def visualize(self, record_fn, img_fn):
        show_agent = False
        show_obstacles = False
        show_routing = True

        reader = RecordReader(record_fn)
        ncolor = 0
        for msg in reader.read_messages():
            if msg.topic == "/apollo/perception/obstacles":
                ncolor += 1

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)

        colorVals = plt.cm.jet(np.linspace(0, 1, ncolor))

        reader = RecordReader(record_fn)
        localization_pb = localization_pb2.LocalizationEstimate()
        is_localization_updated = False
        cnt = 0
        for msg in reader.read_messages():

            if msg.topic == "/apollo/localization/pose":
                if show_agent:
                    localization_pb.ParseFromString(msg.message)
                is_localization_updated = True

            if msg.topic == "/apollo/perception/obstacles":
                if not is_localization_updated:
                    continue

                idx = cnt % ncolor
                c = colorVals[idx]
                cnt += 1

                if show_agent:
                    self.plot_agent(localization_pb, ax, c)
                if show_obstacles:
                    perception_obstacle_pb = perception_obstacle_pb2.PerceptionObstacles()
                    perception_obstacle_pb.ParseFromString(msg.message)
                    for obstacle in perception_obstacle_pb.perception_obstacle:
                        obstacle_plotter.plot(obstacle, ax, c)

            if msg.topic == "/apollo/routing_response":
                if show_routing:
                    routing_response = RoutingResponse()
                    routing_response.ParseFromString(msg.message)
                    routing_plotter.plot(routing_response, ax)

        plt.axis('equal')

        file_utils.makedirs(os.path.dirname(img_fn))
        fig.savefig(img_fn)
        time.sleep(3)
        plt.close(fig)
        time.sleep(1)

    def plot_agent(self, localization_pb, ax, c):
        heading = localization_pb.pose.heading
        position = []
        position.append(localization_pb.pose.position.x)
        position.append(localization_pb.pose.position.y)
        position.append(localization_pb.pose.position.z)
        mkz_plotter.plot(position, heading, ax, c)


if __name__ == "__main__":
    VisualPlanningRecords().main()
