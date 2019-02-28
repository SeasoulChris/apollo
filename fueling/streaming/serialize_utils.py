#!/usr/bin/env python

"""Utility functions for the serialize job"""

import errno
import os
import pickle
import stat
import time

import yaml

from cyber_py import record

import fueling.common.record_utils as record_utils
import fueling.streaming.streaming_utils as streaming_utils

def write_to_file(file_path, mode, message):
    """Write message to file, create new one if not existing"""
    with open(file_path, mode) as file_to_write:
        file_to_write.write(str(message))
    file_stat = os.stat(file_path)
    os.chmod(file_path, file_stat.st_mode | stat.S_IWOTH)

def create_dir_if_not_exist(dir_path):
    """Create directory if it does not exist"""
    if os.path.exists(dir_path):
        return
    try:
        os.makedirs(dir_path)
    except OSError as os_error:
        if os_error.errno != errno.EEXIST:
            raise

def build_file_handles(topic_file_paths):
    """Build a map between topic file and its handle"""
    handles = {}
    for topic_file in topic_file_paths:
        handles[os.path.basename(topic_file)] = open(topic_file, 'a+')
    return handles

def build_line_with_fields(line, fields, message):
    """Combine the fields values with timestamp to form the complete metadata"""
    proto = record_utils.message_to_proto(message)
    field_suffix = '{'
    for field in fields:
        field_parts = field.split('.')
        value = proto
        for part in field_parts:
            value = getattr(value, part)
        field_suffix += field + ':' + value
    field_suffix += '}'
    return '{},{}'.format(line, field_suffix)

def parse_records(records, root_dir):
    """
    Deserialize the record by:
    1. looping messages in the record
    2. parsing and record key meta data in each message, and put to text files partitioned by topic
    3. putting message data itself to a standalone file for quick access
    """
    yaml_file_path = 'fueling/streaming/conf/serialize_conf.yaml'
    settings = list(yaml.load_all(file(yaml_file_path, 'r')))
    for record_file in records:
        record_dir = streaming_utils.record_to_stream_path(record_file, root_dir)
        create_dir_if_not_exist(record_dir)
        topic_files = [os.path.join(record_dir, \
            streaming_utils.topic_to_file_name(x.get('topic'))) for x in settings]
        topic_file_handles = {}
        try:
            topic_file_handles = build_file_handles(topic_files)
            freader = record.RecordReader(record_file)
            for message in freader.read_messages():
                renamed_message = streaming_utils.topic_to_file_name(message.topic)
                if renamed_message in topic_file_handles:
                    line = str(message.timestamp)
                    fields = next(x for x in settings if x.get('topic') == message.topic)\
                        .get('fields')
                    if fields is not None:
                        line = build_line_with_fields(line, fields, message)
                    topic_file_handles[renamed_message].write(line+"\n")
                    message_file_name = '{}-{}'.format(renamed_message, message.timestamp)
                    with open(os.path.join(record_dir, message_file_name), 'wb') as message_file:
                        pickle.dump(message, message_file, pickle.HIGHEST_PROTOCOL)
            write_to_file(os.path.join(record_dir, 'COMPLETE'), 'w', '{:.6f}'.format(time.time()))
        finally:
            for topic in topic_file_handles:
                topic_file_handles[topic].close()
