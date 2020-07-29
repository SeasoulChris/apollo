#!/usr/bin/env python
"""Clean records."""

import os
import resource
import sys
from os import path
from datetime import datetime, timedelta

from cyber.python.cyber_py3.record import RecordWriter
from planning_analytics.cleaner.record_cleaner import RecordCleaner

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class CleanPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.IS_TEST_DATA = False
        self.RUN_IN_DRIVER = True
        now = datetime.now() - timedelta(hours=7)
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        self.dst_prefix = '/mnt/bos/modules/planning/temp/cleaned_data/batch_' + dt_string + "/"

        self.cnt = 1
        self.cleaner = None

    def run_test(self):
        """Run test."""
        record_file = '/fuel/data/20200707105339.record'
        map_file = '/fuel/data/sim_map.bin'

        self.cleaner = RecordCleaner(map_file)
        self.cleaner.process_file(record_file)

        records = ['/fuel/data/broken/']
        # self.to_rdd(records).map(self.process_task).count()
        self.process_task(records[0])

    def run(self):
        """Run prod."""
        task_list_file = path.dirname(path.abspath(__file__)) + "/task_list.txt"
        logging.info(task_list_file)

        tasks_descs = []
        with open(task_list_file, 'r') as f:
            for line in f:
                tasks_descs.append(line)

        if self.RUN_IN_DRIVER:
            for task_desc in tasks_descs:
                self.process_task(task_desc)
        else:
            self.to_rdd(tasks_descs).map(self.process_task).count()

        logging.info('Processing is done')

    def process_task(self, task_desc):
        task_elements = task_desc.replace("\n", "").split(" ")
        task_folder = task_elements[0]
        task_map = task_elements[1]
        map_file = None
        if task_map == "san_mateo":
            map_file = "/mnt/bos/code/baidu/adu-lab/apollo-map/san_mateo/sim_map.bin"
        elif task_map == "sunnyvale_with_two_offices":
            map_file \
                = "/mnt/bos/code/baidu/adu-lab/apollo-map/sunnyvale_with_two_offices/sim_map.bin"

        self.cleaner = RecordCleaner(map_file)

        self.cnt = 1
        files = self.our_storage().list_files(task_folder)
        logging.info('found file num = ' + str(len(files)))

        file_cnt = 0
        total_file_cnt = len(files)

        for fn in files:
            file_cnt += 1
            logging.info("")
            logging.info(
                '[[*]] process file ('
                + str(file_cnt)
                + "/"
                + str(total_file_cnt)
                + "):"
                + fn)

            if record_utils.is_record_file(fn):
                self.cleaner.process_file(fn)
                msgs_list = self.cleaner.get_matured_msg_list()
                for msgs in msgs_list:
                    self.write_msgs(task_folder, msgs, task_map)
                self.cleaner.clean_mature_msg_list()

            if self.cnt > 10 and self.IS_TEST_DATA:
                break

        self.write_msgs(task_folder, self.cleaner.msgs[-1], task_map)

        logging.info("task is done!")

    def write_msgs(self, task_folder, msgs, task_map):

        if len(msgs) < 200 * 10:
            return

        logging.info("write_msgs start: msgs num = " + str(len(msgs)))

        if len(task_folder.split("/")[-1]) == 0:
            task_id = task_folder.split("/")[-2]
        else:
            task_id = task_folder.split("/")[-1]

        dst_record_fn = self.dst_prefix + task_map + "/" + task_id + "/"
        dst_record_fn += str(self.cnt).zfill(5) + ".record"

        self.cnt += 1

        logging.info("Writing output file: " + dst_record_fn)

        file_utils.makedirs(os.path.dirname(dst_record_fn))
        writer = RecordWriter(0, 0)
        try:
            writer.open(dst_record_fn)
            for topic, (data_type, desc) in self.cleaner.topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in msgs:
                if msg is None:
                    print("msg is None")
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(dst_record_fn, e))
            return None
        finally:
            writer.close()
        logging.info("write_msgs end")


def print_current_memory_usage(step_name):
    mb_2_kb = 1024

    meminfo = dict((m.split()[0].rstrip(':'), int(m.split()[1]))

                   for m in open('/proc/meminfo').readlines())

    total_mem = meminfo['MemTotal'] // mb_2_kb

    used_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // mb_2_kb

    logging.info(f'step: {step_name}, total memory: {total_mem} MB, current memory: {used_mem} MB')


if __name__ == '__main__':
    cleaner = CleanPlanningRecords()
    cleaner.main()
