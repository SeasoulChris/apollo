#!/usr/bin/env python
"""Clean records."""

import os

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

    def run_test(self):
        """Run test."""
        self.dst_prefix = '/apollo/data/planning/cleaned_data_temp/'

        records = ['/fuel/fueling/demo/testdata/']
        self.to_rdd(records).map(self.process_folder).count()

    def run(self):
        """Run prod."""

        tasks = [
            'small-records/2019/2019-10-17/2019-10-17-13-36-41/',
            # 'small-records/2018/2018-09-11/2018-09-11-11-10-30/',
        ]

        self.to_rdd(tasks).map(self.process_folder).count()

        logging.info('Processing is done')

    def process_folder(self, task_folder):
        msgs, topic_descs = self.load_msg(task_folder)

        if len(msgs) == 0:
            logging.error('Failed to read any message from {}'.format(task_folder))
            return 0

        routing_analyzer = RoutingUpdateAnalyzer()
        msgs_set, hasrouting_set = routing_analyzer.process(msgs)

        cnt = 1
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
                    self.write_to_file(freq_msgs, topic_descs, task_folder, cnt)
                    cnt += 1

        return 1

    def write_to_file(self, freq_msgs, topic_descs, task_folder, cnt):
        if len(task_folder.split("/")[-1]) == 0:
            task_id = task_folder.split("/")[-2]
        else:
            task_id = task_folder.split("/")[-1]

        dst_record_fn = self.dst_prefix + task_id + "/" + str(cnt).zfill(5) + ".record"
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

    def load_msg(self, task_folder):
        files = self.our_storage().list_files(task_folder)
        logging.info('found file num = ' + str(len(files)))
        msgs = []
        topic_descs = {}
        for fn in files:
            if record_utils.is_record_file(fn):
                try:
                    reader = RecordReader(fn)
                    for msg in reader.read_messages():
                        if msg.topic in self.topics:
                            if msg.topic not in topic_descs:
                                topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
                            msgs.append(msg)
                except Exception as err:
                    continue
        logging.info("total {} original msg".format(len(msgs)))
        return msgs, topic_descs


if __name__ == '__main__':
    cleaner = CleanPlanningRecords()
    cleaner.main()
