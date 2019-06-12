#!/usr/bin/env python

"""Utility functions for the serialize job"""

import os
import time

import colored_glog as glog
import yaml

from cyber_py import record

import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.streaming.streaming_utils as streaming_utils

def build_file_handles(topic_file_paths):
    """Build a map between topic file and its handle"""
    handles = {}
    for topic_file in topic_file_paths:
        func = lambda file_path: open(file_path, 'w+')
        handles[os.path.basename(topic_file)] = streaming_utils.retry(func, [topic_file], 3)
    return handles

def build_meta_with_fields(fields, message):
    """Combine the fields values with timestamp to form the complete metadata"""
    proto = record_utils.message_to_proto(message)
    header_time = int(round(proto.header.timestamp_sec * (10 ** 9)))
    if not fields:
        return header_time, str(header_time)
    field_suffix = '{'
    for field in fields:
        field_parts = field.split('.')
        value = proto
        for part in field_parts:
            value = getattr(value, part)
        field_suffix += '"{}":"{}"'.format(field, value)
    field_suffix += '}'
    return header_time, '{},{}'.format(header_time, field_suffix)

def parse_record(record_file, root_dir):
    """
    Deserialize the record by:
    1. looping messages in the record
    2. parsing and record key meta data in each message, and put to text files partitioned by topic
    3. putting message data itself to a standalone file for quick access
    """
    yaml_file_path = '{}/{}/{}/{}'.format(root_dir,
                                          streaming_utils.STREAMING_PATH,
                                          streaming_utils.STREAMING_CONF, 'serialize_conf.yaml')
    settings = list(yaml.load_all(file(yaml_file_path, 'r')))
    record_file = record_file.strip()
    glog.info('Executor: processsing record file : {}'.format(record_file))
    if not record_utils.is_record_file(record_file):
        return
    record_dir = streaming_utils.record_to_stream_path(record_file,
                                                       root_dir,
                                                       streaming_utils.STREAMING_DATA)
    glog.info('Executor: record directory : {}'.format(record_dir))
    file_utils.makedirs(record_dir)
    if os.path.exists(os.path.join(record_dir, 'COMPLETE')):
        glog.info('target has been generated, do nothing')
        return
    topic_files = [os.path.join(record_dir, \
        streaming_utils.topic_to_file_name(x.get('topic'))) for x in settings]
    topic_file_handles = {}
    try:
        topic_file_handles = build_file_handles(topic_files)
        freader = record.RecordReader(record_file)
        for message in freader.read_messages():
            renamed_topic = streaming_utils.topic_to_file_name(message.topic)
            if renamed_topic in topic_file_handles:
                fields = next(x for x in settings if x.get('topic') == message.topic)\
                         .get('fields')
                header_time, meta = build_meta_with_fields(fields, message)
                topic_file_handles[renamed_topic].write('{}\n'.format(meta))
                streaming_utils.write_message_obj(record_dir, renamed_topic, message, header_time)
        glog.info('completed serializing record file {}.'.format(record_file))
        streaming_utils.write_to_file(os.path.join(record_dir, 'COMPLETE'),
                                      'w',
                                      '{:.6f}'.format(time.time()))
    finally:
        for topic in topic_file_handles:
            topic_file_handles[topic].close()
