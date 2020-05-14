#!/usr/bin/env python
"""Clean records."""

import os
from datetime import datetime, timedelta
import time
import resource

from cyber_py3.record import RecordReader, RecordWriter
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

from fueling.common.base_pipeline import BasePipeline
from fueling.planning.apollo_record_reader.apollo_record_reader import ApolloRecordReader
from fueling.planning.cleaner.analyzer_routing import RoutingAnalyzer
from fueling.planning.cleaner.analyzer_hmi import HmiAnalyzer
from fueling.planning.cleaner.analyzer_localization import LocalizationAnalyzer
from fueling.planning.cleaner.analyzer_chassis import ChassisAnalyzer
from fueling.planning.cleaner.analyzer_perception import PerceptionAnalyzer

from fueling.planning.cleaner.analyzer_prediction import PredictionAnalyzer


class CleanPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.IS_TEST_DATA = False
        self.RUN_IN_DRIVER = False
        now = datetime.now() - timedelta(hours=7)
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        self.dst_prefix = '/mnt/bos/modules/planning/temp/cleaned_data/batch_' + dt_string + "/"
        self.topics = [
            '/apollo/canbus/chassis',
            '/apollo/localization/pose',
            '/apollo/hmi/status',
            '/apollo/perception/obstacles',
            '/apollo/perception/traffic_light',
            '/apollo/prediction',
            '/apollo/routing_request',
            '/apollo/routing_response',
            '/apollo/routing_response_history',
        ]
        self.cnt = 1
        self.msgs = list()
        self.topic_descs = {}
        self.current_hmi_status_msg = None
        self.current_routing_response = None
        self.has_routing = False

        self.routing_analyzer = None
        self.hmi_analyzer = None

        self.localization_analyzer = None
        self.chassis_analyzer = None

        self.perception_analyzer = None
        self.prediction_analyzer = None

    def run_test(self):
        """Run test."""
        self.dst_prefix = '/fuel/data/planning/cleaned_data_temp/'

        records = ['/fuel/data/broken/']
        # self.to_rdd(records).map(self.process_task).count()
        self.process_task(records[0])

    def run(self):
        """Run prod."""
        date_tasks = [
            #'small-records/2019/2019-11-01/',
            #'small-records/2019/2019-11-02/',
            #'small-records/2019/2019-11-03/',
            #'small-records/2019/2019-11-04/',
            #'small-records/2019/2019-11-05/',
            #'small-records/2019/2019-11-06/',
            #'small-records/2019/2019-11-07/',
            #'small-records/2019/2019-11-08/',
            #'small-records/2019/2019-11-09/',
            #'small-records/2019/2019-11-10/',

            # 'small-records/2019/2019-11-11/',
            # 'small-records/2019/2019-11-12/',
            # 'small-records/2019/2019-11-13/',
            # 'small-records/2019/2019-11-14/',
            # 'small-records/2019/2019-11-15/',
            # 'small-records/2019/2019-11-16/',
            # 'small-records/2019/2019-11-17/',
            # 'small-records/2019/2019-11-18/',
            # 'small-records/2019/2019-11-19/',
            # 'small-records/2019/2019-11-20/',
            # 'small-records/2019/2019-11-21/',
            # 'small-records/2019/2019-11-22/',
            # 'small-records/2019/2019-11-23/',
            # 'small-records/2019/2019-11-24/',
            # 'small-records/2019/2019-11-25/',
            # 'small-records/2019/2019-11-26/',
            # 'small-records/2019/2019-11-27/',
            # 'small-records/2019/2019-11-28/',
            # 'small-records/2019/2019-11-29/',
            # 'small-records/2019/2019-11-30/',
        ]

        individual_tasks = [
            # 'small-records/2019/2019-10-17/2019-10-17-13-36-41/',
            # 'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
            'modules/planning/temp/converted_data_with_routing/batch_20200513_172433/MKZ173_20200121122216',
            'modules/planning/temp/converted_data_with_routing/batch_20200513_172433/MKZ170_20200121120310',
            'modules/planning/temp/converted_data_with_routing/batch_20200513_172433/MKZ167_20200121131624'
        ]
        prefix = "/mnt/bos/"

        final_tasks = []
        for day_folder in date_tasks:
            day_abs_folder = prefix + day_folder
            if not os.path.exists(day_abs_folder):
                continue
            for filename in os.listdir(day_abs_folder):
                file_path = os.path.join(day_abs_folder, filename)
                try:
                    if os.path.isdir(file_path):
                        final_task = file_path.replace(prefix, "")
                        logging.info("found a task: " + final_task)
                        final_tasks.append(final_task)

                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        for task in individual_tasks:
            final_tasks.append(task)

        if self.RUN_IN_DRIVER:
            for task in final_tasks:
                self.process_task(task)
        else:
            self.to_rdd(final_tasks).map(self.process_task).count()

        logging.info('Processing is done')

    def process_task(self, task_folder):
        self.cnt = 1
        files = self.our_storage().list_files(task_folder)
        logging.info('found file num = ' + str(len(files)))

        del self.msgs
        self.msgs = list()
        self.topic_descs = dict()

        self.routing_analyzer = RoutingAnalyzer()
        self.hmi_analyzer = HmiAnalyzer()

        self.localization_analyzer = LocalizationAnalyzer()
        self.chassis_analyzer = ChassisAnalyzer()

        self.perception_analyzer = PerceptionAnalyzer()
        self.prediction_analyzer = PredictionAnalyzer()

        file_cnt = 0
        total_file_cnt = len(files)

        for fn in files:
            file_cnt += 1
            logging.info("")
            logging.info(
                '[[*]] process file (' +
                str(file_cnt) +
                "/" +
                str(total_file_cnt) +
                "):" +
                fn)

            if record_utils.is_record_file(fn):
                self.process_file2(fn, task_folder)

            # TODO this is for testing data
            if self.cnt > 10 and self.IS_TEST_DATA:
                break

        self.write_msgs(task_folder)

        logging.info("total {} original msg".format(len(self.msgs)))

    def process_file2(self, filename, task_folder):
        print_current_memory_usage("ProcessFile-before")

        reader = ApolloRecordReader()
        logging.info("start reading messages...")
        for msg in reader.read_messages(filename):
            if msg.topic not in self.topics:
                continue

            if msg.topic == '/apollo/routing_response':
                if self.routing_analyzer.get_routing_response_msg() is not None:
                    self.write_msgs(task_folder)

                self.routing_analyzer.update(msg)

            if msg.topic == '/apollo/routing_response_history':
                if self.routing_analyzer.get_routing_response_msg() is None:
                    self.routing_analyzer.update(msg)

            if msg.topic == "/apollo/hmi/status":
                self.hmi_analyzer.update(msg)

            if msg.topic == '/apollo/perception/obstacles':

                perception_timestamp = PerceptionAnalyzer.get_msg_timstamp(msg)
                last_perception_timestamp = self.perception_analyzer.get_last_perception_timestamp()
                last_chassis_timestamp = self.chassis_analyzer.get_last_chassis_timestamp()
                last_localization_timestamp = self.localization_analyzer.get_last_localization_timestamp()
                last_prediction_timestamp = self.prediction_analyzer.get_last_prediction_timestamp()

                if abs(perception_timestamp - last_chassis_timestamp) > 0.05 \
                        or abs(perception_timestamp - last_localization_timestamp) > 0.05 \
                        or abs(perception_timestamp - last_prediction_timestamp) > 0.5 \
                        or abs(perception_timestamp - last_perception_timestamp) > 0.5:
                    # logging.info("Some msg is missing! ")
                    self.write_msgs(task_folder)

                    routing_msg = self.routing_analyzer.get_routing_response_msg()
                    if routing_msg is not None:
                        self.msgs.append(routing_msg)
                        if self.hmi_analyzer.get_hmi_status_msg() is not None:
                            self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())

                self.perception_analyzer.update(msg)

            if msg.topic == '/apollo/prediction':
                self.prediction_analyzer.update(msg)

            if msg.topic == '/apollo/localization/pose':
                localization = self.localization_analyzer.get_localization_estimate(msg)
                if self.routing_analyzer.get_routing_response_msg() is not None and \
                        not self.routing_analyzer.is_adv_on_routing(localization):
                    # logging.info("ADV is not on routing! ")
                    self.write_msgs(task_folder)

                    self.msgs.append(self.routing_analyzer.get_routing_response_msg())
                    if self.hmi_analyzer.get_hmi_status_msg() is not None:
                        self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())

                self.localization_analyzer.update(msg)

            if msg.topic == '/apollo/canbus/chassis':
                self.chassis_analyzer.update(msg)

            if self.routing_analyzer.get_routing_response_msg() is not None:
                self.msgs.append(msg)

            if len(self.msgs) >= 200 * 60 * 5:
                self.write_msgs(task_folder)

                self.msgs.append(self.routing_analyzer.get_routing_response_msg())
                if self.hmi_analyzer.get_hmi_status_msg() is not None:
                    self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())

        channels = reader.get_channels()
        for channel in channels:
            self.topic_descs[channel.name] = (channel.message_type, channel.proto_desc)

        print_current_memory_usage("ProcessFile-after")
        logging.info("")

    def write_msgs(self, task_folder):

        if len(self.msgs) < 200 * 10:
            # logging.info("len(self.msgs) < 200 * 10 ... " + "write_msgs end")
            del self.msgs
            self.msgs = list()
            # logging.info("write_msgs end")
            return

        logging.info("write_msgs start: msgs num = " + str(len(self.msgs)))

        if len(task_folder.split("/")[-1]) == 0:
            task_id = task_folder.split("/")[-2]
        else:
            task_id = task_folder.split("/")[-1]

        dst_record_fn = self.dst_prefix + task_id + "/" + str(self.cnt).zfill(5) + ".record"
        self.cnt += 1

        logging.info("Writing output file: " + dst_record_fn)
        # Write to record.
        file_utils.makedirs(os.path.dirname(dst_record_fn))
        writer = RecordWriter(0, 0)
        try:
            writer.open(dst_record_fn)
            for topic, (data_type, desc) in self.topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in self.msgs:
                if msg is None:
                    print("msg is None")
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(dst_record_fn, e))
            return None
        finally:
            writer.close()
            del self.msgs
            self.msgs = list()
        logging.info("write_msgs end")

    def process_file(self, filename, task_folder):

        print_current_memory_usage("ProcessFile-before")

        reader = RecordReader(filename)
        time.sleep(5)
        logging.info("start reading messages...")
        for msg in reader.read_messages():
            if msg.topic not in self.topics:
                continue

            if msg.topic not in self.topic_descs:
                self.topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))

            if msg.topic == '/apollo/routing_response':
                if self.routing_analyzer.get_routing_response_msg() is not None:
                    self.write_msgs(task_folder)

                self.routing_analyzer.update(msg)

            if msg.topic == '/apollo/routing_response_history':
                if self.routing_analyzer.get_routing_response_msg() is None:
                    self.routing_analyzer.update(msg)

            if msg.topic == "/apollo/hmi/status":
                self.hmi_analyzer.update(msg)

            if msg.topic == '/apollo/perception/obstacles':

                perception_timestamp = PerceptionAnalyzer.get_msg_timstamp(msg)
                last_perception_timestamp = self.perception_analyzer.get_last_perception_timestamp()
                last_chassis_timestamp = self.chassis_analyzer.get_last_chassis_timestamp()
                last_localization_timestamp = self.localization_analyzer.get_last_localization_timestamp()
                last_prediction_timestamp = self.prediction_analyzer.get_last_prediction_timestamp()

                if abs(perception_timestamp - last_chassis_timestamp) > 0.05 \
                        or abs(perception_timestamp - last_localization_timestamp) > 0.05 \
                        or abs(perception_timestamp - last_prediction_timestamp) > 0.5 \
                        or abs(perception_timestamp - last_perception_timestamp) > 0.5:
                    # logging.info("Some msg is missing! ")
                    self.write_msgs(task_folder)
                    if self.routing_analyzer.get_routing_response_msg() is not None:
                        self.msgs.append(self.routing_analyzer.get_routing_response_msg())
                        if self.hmi_analyzer.get_hmi_status_msg() is not None:
                            self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())
                if self.msgs[-1] is None:
                    print("/apollo/perception/obstacles  is None")

                self.perception_analyzer.update(msg)

            if msg.topic == '/apollo/prediction':
                self.prediction_analyzer.update(msg)

            if msg.topic == '/apollo/localization/pose':
                localization = self.localization_analyzer.get_localization_estimate(msg)
                if self.routing_analyzer.get_routing_response_msg() is not None and \
                        not self.routing_analyzer.is_adv_on_routing(localization):
                    # logging.info("ADV is not on routing! ")
                    self.write_msgs(task_folder)

                    self.msgs.append(self.routing_analyzer.get_routing_response_msg())
                    if self.hmi_analyzer.get_hmi_status_msg() is not None:
                        self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())

                if self.msgs[-1] is None:
                    print("/apollo/localization/pose  is None")
                self.localization_analyzer.update(msg)

            if msg.topic == '/apollo/canbus/chassis':
                self.chassis_analyzer.update(msg)

            if self.routing_analyzer.get_routing_response_msg() is not None:
                self.msgs.append(msg)

            if len(self.msgs) >= 200 * 60 * 5:
                self.write_msgs(task_folder)

                self.msgs.append(self.routing_analyzer.get_routing_response_msg())
                if self.hmi_analyzer.get_hmi_status_msg() is not None:
                    self.msgs.append(self.hmi_analyzer.get_hmi_status_msg())

        reader.reset()
        del reader
        time.sleep(5)
        logging.info("deleted reader")

        print_current_memory_usage("ProcessFile-after")
        logging.info("")


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
