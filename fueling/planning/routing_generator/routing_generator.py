#!/usr/bin/env python
"""Clean records."""

import os
import resource
import sys
from datetime import datetime, timedelta
from os import path

sys.path.append('/fuel/fueling/planning/analytics/planning_analytics.zip')
sys.path.append('fueling/planning/analytics/planning_analytics.zip')
sys.path.append(path.dirname(path.abspath(__file__)) + "/../analytics/planning_analytics.zip")

from cyber_py3.record import RecordWriter
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.file_utils as file_utils
from fueling.common.base_pipeline import BasePipeline

from planning_analytics.apl_record_reader.apl_record_reader import AplRecordReader
from planning_analytics.route_generator.route_generator import RouteGenerator


class RoutingGenerator(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        now = datetime.now() - timedelta(hours=7)
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        self.dst_prefix = ('/mnt/bos/modules/planning/temp/converted_data_with_routing/batch_'
                           + dt_string + "/")
        self.map_file = '/mnt/bos/code/baidu/adu-lab/apollo-map/yizhuangdaluwang/sim_map.bin'
        self.apl_topics = [
            '/apollo/canbus/chassis',
            '/apollo/localization/pose',
            # '/apollo/hmi/status',
            '/apollo/perception/obstacles',
            '/apollo/perception/traffic_light',
            '/apollo/prediction',
            # '/apollo/routing_request',
            '/apollo/routing_response',
            # '/apollo/routing_response_history',
        ]

        self.topic_descs = dict()

        self.RUN_IN_DRIVER = True

    def run_test(self):
        """Run test."""
        self.dst_prefix = '/fuel/data/planning/temp/converted_data_with_routing/'

        individual_tasks = ['/fuel/data/broken/']
        # self.to_rdd(records).map(self.process_task).count()
        # self.process_task(records[0])

        task_files = []
        for task in individual_tasks:
            files = self.our_storage().list_files(task)
            for file in files:
                if record_utils.is_record_file(file):
                    task_files.append(file)

        for file in task_files:
            logging.info(file)
            self.process_file(file)

    def run(self):
        """Run prod."""
        individual_tasks = [
            'modules/planning/temp/converted_data/batch_20200512_171650/MKZ173_20200121122216',
            'modules/planning/temp/converted_data/batch_20200512_171650/MKZ170_20200121120310',
            'modules/planning/temp/converted_data/batch_20200512_171650/MKZ167_20200121131624',
        ]

        task_files = []
        for task in individual_tasks:
            files = self.our_storage().list_files(task)
            for file in files:
                if record_utils.is_record_file(file):
                    task_files.append(file)

        if self.RUN_IN_DRIVER:
            for file in task_files:
                self.process_file(file)
        else:
            self.to_rdd(task_files).map(self.process_file).count()

        logging.info('Processing is done')

    def process_file(self, record_path_file):
        logging.info("")
        logging.info("* input_file: " + record_path_file)

        route_generator = RouteGenerator(self.map_file)
        route_response_msg = route_generator.generate(record_path_file)

        self.get_topic_descs()

        record_file = record_path_file.split(os.sep)[-1]
        record_task = record_path_file.split(os.sep)[-2]

        output_record_file = self.dst_prefix + record_task + os.sep + record_file
        logging.info("output_file: " + output_record_file)

        file_utils.makedirs(os.path.dirname(output_record_file))

        reader = AplRecordReader()
        writer = RecordWriter(0, 0)
        try:
            writer.open(output_record_file)

            logging.info("writing channels.")
            for topic, (data_type, desc) in self.topic_descs.items():
                writer.write_channel(topic, data_type, desc)

            logging.info("writing msgs.")
            routing_written = False
            for msg in reader.read_messages(record_path_file):
                # if msg.topic == '/apollo/routing_request':
                if not routing_written:
                    writer.write_message('/apollo/routing_response',
                                         route_response_msg.SerializeToString(),
                                         msg.timestamp - 1)
                    routing_written = True

                writer.write_message(msg.topic, msg.message, msg.timestamp)

        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(output_record_file, e))
            return None
        finally:
            writer.close()

    def get_topic_descs(self):
        filename = ("/mnt/bos/small-records/2019/2019-11-11/"
                    + "2019-11-11-06-24-26/20191111062526.record")
        reader = AplRecordReader()
        for msg in reader.read_messages(filename):
            continue

        channels = reader.get_channels()
        for channel in channels:
            self.topic_descs[channel.name] = (channel.message_type, channel.proto_desc)


def print_current_memory_usage(step_name):
    mb_2_kb = 1024

    meminfo = dict((m.split()[0].rstrip(':'), int(m.split()[1]))

                   for m in open('/proc/meminfo').readlines())

    total_mem = meminfo['MemTotal'] // mb_2_kb

    used_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // mb_2_kb

    logging.info(f'step: {step_name}, total memory: {total_mem} MB, current memory: {used_mem} MB')


if __name__ == '__main__':
    cleaner = RoutingGenerator()
    cleaner.main()
