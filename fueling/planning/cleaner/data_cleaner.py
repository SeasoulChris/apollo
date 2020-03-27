#!/usr/bin/env python
"""Clean records."""

import os
import time

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from cyber_py3.record import RecordReader, RecordWriter
from fueling.common.base_pipeline import BasePipeline
from fueling.planning.cleaner.msg_freq_analyzer import MsgFreqAnalyzer
from fueling.planning.cleaner.routing_update_analyzer import RoutingUpdateAnalyzer


class CleanPlanningRecords(BasePipeline):
    """CleanPlanningRecords pipeline."""

    def __init__(self):
        self.metrics_prefix = 'data.pipelines.clean_planning_records.'
        self.dst_prefix = '/mnt/bos/modules/planning/cleaned_data_temp/'
        self.src_prefixs = [
            # 'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
            'small-records/2020/2020-02-19/2020-02-19-15-10-48/'
        ]
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

    def run_test(self):
        """Run test."""
        self.dst_prefix = '/apollo/data/planning/cleaned_data_temp/'

        records = ['/fuel/data/task1/']
        self.to_rdd(records).map(self.process_folder).count()

    def run(self):
        """Run prod."""

        tasks = [
            # 'small-records/2019/2019-11-01/',
            'small-records/2019/2019-11-02/',
            'small-records/2019/2019-11-03/',
            'small-records/2019/2019-11-04/',
            'small-records/2019/2019-11-05/',
            'small-records/2019/2019-11-06/',
            'small-records/2019/2019-11-07/',
            'small-records/2019/2019-11-08/',
            'small-records/2019/2019-11-09/',
            'small-records/2019/2019-11-10/',
            'small-records/2019/2019-11-11/',
            'small-records/2019/2019-11-12/',
            'small-records/2019/2019-11-13/',
            'small-records/2019/2019-11-14/',
            'small-records/2019/2019-11-15/',
            'small-records/2019/2019-11-16/',
            'small-records/2019/2019-11-17/',
            'small-records/2019/2019-11-18/',
            'small-records/2019/2019-11-19/',
            'small-records/2019/2019-11-20/',
            'small-records/2019/2019-11-21/',
            'small-records/2019/2019-11-22/',
            'small-records/2019/2019-11-23/',
            'small-records/2019/2019-11-24/',
            'small-records/2019/2019-11-25/',
            'small-records/2019/2019-11-26/',
            'small-records/2019/2019-11-27/',
            'small-records/2019/2019-11-28/',
            'small-records/2019/2019-11-29/',
            'small-records/2019/2019-11-30/',
            # 'small-records/2019/2019-10-17/2019-10-17-13-36-41/',
            # 'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
        ]

        prefix = "/mnt/bos/"

        final_tasks = []
        for day_folder in tasks:
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

        self.to_rdd(final_tasks).map(self.process_folder_ecom).count()

        logging.info('Processing is done')

    def process_folder_ecom(self, task_folder):
        self.cnt = 1
        files = self.our_storage().list_files(task_folder)
        logging.info('found file num = ' + str(len(files)))
        msgs = []
        topic_descs = {}
        has_routing = False

        for fn in files:
            logging.info('process file: ' + fn)

            if record_utils.is_record_file(fn):
                try:
                    reader = RecordReader(fn)
                    time.sleep(2)
                    for msg in reader.read_messages():
                        if msg.topic == '/apollo/routing_response':
                            logging.info("found a routing response!")
                            has_routing = True
                            self.process_msgs(msgs, task_folder, topic_descs)
                            msgs = []

                        if msg.topic in self.topics:
                            if msg.topic not in topic_descs:
                                topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
                            if has_routing:
                                msgs.append(msg)
                    del reader
                except Exception as err:
                    print(err)
                    continue

        logging.info("total {} original msg".format(len(msgs)))

    def process_msgs(self, msgs, task_folder, topic_descs):
        if len(msgs) == 0:
            logging.error('Failed to read any message from {}'.format(task_folder))
            return 0

        routing_analyzer = RoutingUpdateAnalyzer()
        msgs_set, hasrouting_set = routing_analyzer.process(msgs)

        for i in range(len(msgs_set)):
            same_routing_msgs = msgs_set[i]
            has_routing = hasrouting_set[i]
            routing_resp = None
            hmi_status = None
            if same_routing_msgs[0].topic == '/apollo/routing_response':
                routing_resp = same_routing_msgs[0]
            if same_routing_msgs[0].topic == "/apollo/hmi/status":
                hmi_status = same_routing_msgs[0]
            if same_routing_msgs[1].topic == '/apollo/routing_response':
                routing_resp = same_routing_msgs[1]
            if same_routing_msgs[1].topic == "/apollo/hmi/status":
                hmi_status = same_routing_msgs[1]

            logging.info("Has Routing = " + str(has_routing))

            if has_routing:
                freq_analyzer = MsgFreqAnalyzer()
                freq_msgs_set = freq_analyzer.process(same_routing_msgs)
                for freq_msgs in freq_msgs_set:
                    if freq_msgs[0].topic != '/apollo/hmi/status':
                        if hmi_status is not None:
                            freq_msgs.insert(0, hmi_status)

                    if freq_msgs[0].topic != '/apollo/routing_response' \
                            and freq_msgs[1].topic != '/apollo/routing_response':
                        if routing_resp is not None:
                            freq_msgs.insert(0, routing_resp)
                    self.write_to_file(freq_msgs, topic_descs, task_folder)

        return 1

    def write_to_file(self, freq_msgs, topic_descs, task_folder):
        if len(task_folder.split("/")[-1]) == 0:
            task_id = task_folder.split("/")[-2]
        else:
            task_id = task_folder.split("/")[-1]

        dst_record_fn = self.dst_prefix + task_id + "/" + str(self.cnt).zfill(5) + ".record"
        self.cnt += 1

        logging.info(dst_record_fn)
        # Write to record.
        file_utils.makedirs(os.path.dirname(dst_record_fn))
        writer = RecordWriter(0, 0)
        try:
            writer.open(dst_record_fn)
            for topic, (data_type, desc) in topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in freq_msgs:
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(dst_record_fn, e))
            return None
        finally:
            writer.close()
        return dst_record_fn


if __name__ == '__main__':
    cleaner = CleanPlanningRecords()
    cleaner.main()
