#!/usr/bin/env python
"""Clean records."""

import os
from datetime import datetime, timedelta
from os import path

import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from fueling.common.base_pipeline import BasePipeline
from planning_analytics.nudge.record_nudge_analyzer import RecordNudgeAnalzyer


class CleanPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.IS_TEST_DATA = False
        self.RUN_IN_DRIVER = True
        now = datetime.now() - timedelta(hours=7)
        self.dt_string = now.strftime("%Y%m%d_%H%M%S")
        self.dst_prefix \
            = '/mnt/bos/modules/planning/temp/nudge_detection/' \
              + self.dt_string + "_nudges_detected.txt"

        if not os.path.exists(os.path.dirname(self.dst_prefix)):
            try:
                os.makedirs(os.path.dirname(self.dst_prefix))
            except OSError:  # Guard against race condition
                pass

        self.cnt = 1
        self.detector = None

    def run_test(self):
        """Run test."""
        pass

    def _get_task_descs(self):
        task_list_file = path.dirname(path.abspath(__file__)) + "/task_list.txt"
        logging.info(task_list_file)

        tasks_descs = []
        tasks_map_dict = dict()
        with open(task_list_file, 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                line_elements = line.split(" ")
                folder = line_elements[0]
                task_map = line_elements[1]

                files = self.our_storage().list_files(folder)
                for fn in files:
                    if record_utils.is_record_file(fn):
                        task = "/".join(fn.split("/")[3:-1])
                        if task not in tasks_map_dict:
                            tasks_map_dict[task] = task_map

        for task, task_map in tasks_map_dict.items():
            task_desc = task + " " + task_map
            logging.info(task_desc)
            tasks_descs.append(task_desc)

        return tasks_descs

    def _get_task_map_file(self, task_map):
        map_file = ""
        if task_map == "san_mateo":
            map_file = "/mnt/bos/code/baidu/adu-lab/apollo-map/san_mateo/sim_map.bin"
        elif task_map == "sunnyvale_with_two_offices":
            map_file \
                = "/mnt/bos/code/baidu/adu-lab/apollo-map/sunnyvale_with_two_offices/sim_map.bin"
        elif task_map == "yizhuangdaluwang":
            map_file \
                = "/mnt/bos/code/baidu/adu-lab/apollo-map/yizhuangdaluwang/sim_map.bin"
        return map_file

    def run(self):
        """Run prod."""
        tasks_descs = self._get_task_descs()

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
        map_file = self._get_task_map_file(task_map)

        self.detector = RecordNudgeAnalzyer()

        self.cnt = 1

        files = self.our_storage().list_files(task_folder)
        logging.info('found file num = ' + str(len(files)))

        file_cnt = 0
        total_file_cnt = len(files)

        for fn in files:
            file_cnt += 1
            logging.info("")
            logging.info(
                '[[*]] process file (' + str(file_cnt) + "/"
                + str(total_file_cnt) + "):" + fn)

            if record_utils.is_record_file(fn):
                timestamps = self.detector.process(fn, map_file)
                if len(timestamps) > 0:
                    with open(self.dst_prefix, 'a') as f:
                        f.write(fn + "\n")
                        for timestamp in timestamps:
                            logging.info("found a nudge at " + str(timestamp))
                            f.write(str(timestamp) + "\n")
                        f.write("\n")

        logging.info("task is done!")


if __name__ == '__main__':
    cleaner = CleanPlanningRecords()
    cleaner.main()
