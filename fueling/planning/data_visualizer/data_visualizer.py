#!/usr/bin/env python3

import datetime
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    from cyber_py3.record import RecordReader
else:
    from cyber_py.record import RecordReader
from modules.localization.proto import localization_pb2
from modules.perception.proto import perception_obstacle_pb2

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from fueling.common.base_pipeline import BasePipeline

from fueling.planning.data_visualizer import mkz_plotter
from fueling.planning.data_visualizer import obstacle_plotter


class VisualPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.data_version = "ver_20200219_213417"

        self.metrics_prefix = 'data.pipelines.visual_planning_records.'
        self.dst_prefix = 'modules/planning/visual_data/' + self.data_version + "/"
        self.src_prefixs = [
            'modules/planning/cleaned_data/' + self.data_version + '/',
        ]

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        self.dst_prefix = '/fuel/data/planning/visual_data/ver_' \
                          + datetime.date.today().strftime("%Y%m%d_%H%M%S") + "/"

        records = ['/fuel/testdata/data/small.record', '/fuel/testdata/data/small.record']
        for record in records:
            self.process_record(record)

    def run_prod(self):
        """Run prod."""

        # RDD(record_path)
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix)).filter(record_utils.is_record_file)
            for prefix in self.src_prefixs])

        processed_records = records_rdd.map(self.process_record)
        logging.info('Processed {} records'.format(processed_records.count()))

    def process_record(self, src_record_fn):
        src_record_fn_elements = src_record_fn.split("/")
        task_id = src_record_fn_elements[-2]
        fn = src_record_fn_elements[-1]

        dst_record_fn_elements = src_record_fn_elements[:-6]
        dst_record_fn_elements.append(self.dst_prefix)
        dst_record_fn_elements.append(task_id)
        dst_record_fn_elements.append(fn)

        dst_img_fn = "/".join(dst_record_fn_elements) + ".pdf"
        logging.info(dst_img_fn)

        self.visualize(src_record_fn, dst_img_fn)

        return dst_img_fn

    def visualize(self, record_fn, img_fn):
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
                localization_pb.ParseFromString(msg.message)
                is_localization_updated = True

            if msg.topic == "/apollo/perception/obstacles":
                if not is_localization_updated:
                    continue

                idx = cnt % ncolor
                c = colorVals[idx]
                cnt += 1

                perception_obstacle_pb = perception_obstacle_pb2.PerceptionObstacles()
                perception_obstacle_pb.ParseFromString(msg.message)

                self.plot_agent(localization_pb, ax, c)
                for obstacle in perception_obstacle_pb.perception_obstacle:
                    obstacle_plotter.plot(obstacle, ax, c)
        plt.axis('equal')

        file_utils.makedirs(os.path.dirname(img_fn))
        fig.savefig(img_fn)
        time.sleep(1)
        plt.close(fig)

    def plot_agent(self, localization_pb, ax, c):
        heading = localization_pb.pose.heading
        position = []
        position.append(localization_pb.pose.position.x)
        position.append(localization_pb.pose.position.y)
        position.append(localization_pb.pose.position.z)
        mkz_plotter.plot(position, heading, ax, c)


if __name__ == "__main__":
    VisualPlanningRecords().main()
