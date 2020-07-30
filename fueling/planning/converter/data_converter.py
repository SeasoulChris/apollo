#!/usr/bin/env python
"""Clean records."""

import os
import resource
import sys
from datetime import datetime, timedelta
from os import path

from cyber.python.cyber_py3.record import RecordWriter
from planning_analytics.record_converter.record_converter import RecordConverter
from planning_analytics.apl_record_reader.apl_record_reader import AplRecordReader

import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.file_utils as file_utils
from fueling.common.base_pipeline import BasePipeline


class DataConverter(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        now = datetime.now() - timedelta(hours=7)
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        self.dst_prefix = '/mnt/bos/modules/planning/temp/converted_data/batch_' + dt_string + "/"
        self.apl_topics = [
            '/apollo/canbus/chassis',
            '/apollo/localization/pose',
            # '/apollo/hmi/status',
            '/apollo/perception/obstacles',
            '/apollo/perception/traffic_light',
            '/apollo/prediction',
            '/apollo/routing_request',
            # '/apollo/routing_response',
            # '/apollo/routing_response_history',
        ]
        self.tlz_topics = [
            '/localization/100hz/localization_pose',
            '/pnc/prediction',
            '/planning/proxy/DuDriveChassis',
            '/perception/traffic_light_trigger',
            '/perception/traffic_light_status',
            '/router/routing_signal',
            '/pnc/decision',
        ]
        self.topic_descs = dict()

        self.RUN_IN_DRIVER = True

    def run_test(self):
        """Run test."""
        self.dst_prefix = '/fuel/data/planning/temp/converted_data/'

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
        task_list_file = path.dirname(path.abspath(__file__)) + "/task_list.txt"
        logging.info(task_list_file)

        task_files = []
        with open(task_list_file, 'r') as f:
            for task_desc in f:
                day_task_folder = task_desc.split(" ")[0]
                task_filter = task_desc.split(" ")[1]
                files = self.our_storage().list_files(day_task_folder)
                for file in files:
                    record_task = file.split(os.sep)[-2]
                    if task_filter not in record_task:
                        continue
                    if record_utils.is_record_file(file):
                        task_files.append(file)

        if self.RUN_IN_DRIVER:
            for file in task_files:
                self.process_file(file)
        else:
            self.to_rdd(task_files).map(self.process_file).count()

        logging.info('Processing is done')

    def process_file(self, tlz_record_file):
        logging.info("")
        logging.info("* input_file: " + tlz_record_file)
        self.get_topic_descs()

        record_file = tlz_record_file.split(os.sep)[-1]
        record_task = tlz_record_file.split(os.sep)[-2]

        output_record_file = self.dst_prefix + record_task + os.sep + record_file
        logging.info("output_file: " + output_record_file)

        file_utils.makedirs(os.path.dirname(output_record_file))

        converter = RecordConverter()
        writer = RecordWriter(0, 0)
        try:
            writer.open(output_record_file)

            logging.info("writing channels.")
            for topic, (data_type, desc) in self.topic_descs.items():
                writer.write_channel(topic, data_type, desc)

            logging.info("writing msgs.")
            for msg in converter.convert(tlz_record_file):
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

    def get_individual_tasks(self):
        individual_tasks = [
            'modules/data/planning/MKZ167_20200121131624/',
            'modules/data/planning/MKZ170_20200121120310/',
            'modules/data/planning/MKZ173_20200121122216/',
        ]

        task_files = []
        for task in individual_tasks:
            files = self.our_storage().list_files(task)
            for file in files:
                if record_utils.is_record_file(file):
                    task_files.append(file)

        return task_files


def print_current_memory_usage(step_name):
    mb_2_kb = 1024

    meminfo = dict((m.split()[0].rstrip(':'), int(m.split()[1]))

                   for m in open('/proc/meminfo').readlines())

    total_mem = meminfo['MemTotal'] // mb_2_kb

    used_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // mb_2_kb

    logging.info(f'step: {step_name}, total memory: {total_mem} MB, current memory: {used_mem} MB')


if __name__ == '__main__':
    cleaner = DataConverter()
    cleaner.main()
