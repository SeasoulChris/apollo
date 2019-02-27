#!/usr/bin/env python

"""Utility functions for streaming jobs"""

import errno
import os
import pickle
import stat

import yaml

from cyber_py import record

import fueling.common.record_utils as record_utils

def write_to_file(file_path, mode, message):
    """Write message to file, create new one if not existing"""
    with open(file_path, mode) as file_to_write:
        file_to_write.write(message)
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
    filed_suffix = '{'
    for field in fields:
        field_parts = field.split('.')
        value = proto
        for part in field_parts:
            value = getattr(value, part)
        filed_suffix += field + ':' + value
    filed_suffix += '}'
    return '{},{}'.format(line, filed_suffix)

def topic_to_file(topic):
    """Convert / in topic to - to make it compatible with file system"""
    return topic.replace('/', '_')

def parse_records(records, root_dir, streaming_dir):
    """
    Deserialize the record by:
    1. looping messages in the record
    2. parsing and record key meta data in each message, and put to text files partitioned by topic
    3. putting message data itself to a standalone file for quick access
    """
    target_dir = os.path.join(root_dir, os.path.join(streaming_dir, 'data'))
    yaml_file_path = 'fueling/streaming/conf/serialize_conf.yaml'
    settings = yaml.load_all(file(yaml_file_path, 'r'))
    record_key = '2019'
    for record_file in records:
        record_dir = os.path.join(target_dir, record_file[record_file.find(record_key):])
        create_dir_if_not_exist(record_dir)
        topic_files = [os.path.join(record_dir, topic_to_file(x.get('topic'))) for x in settings]
        topic_file_handles = {}
        try:
            topic_file_handles = build_file_handles(topic_files)
            freader = record.RecordReader(record_file)
            for message in freader.read_messages():
                renamed_message = topic_to_file(message.topic)
                if renamed_message in topic_file_handles:
                    line = str(message.timestamp)
                    fields = next((x for x in settings if x.get('topic') == message.topic), None)
                    if fields is not None:
                        line = build_line_with_fields(line, fields, message)
                    topic_file_handles[renamed_message].write(line+"\n")
                    message_file_name = '{}-{}'.format(renamed_message, message.timestamp)
                    with open(os.path.join(record_dir, message_file_name), 'wb') as message_file:
                        pickle.dump(message, message_file, pickle.HIGHEST_PROTOCOL)
        finally:
            for topic in topic_file_handles:
                topic_file_handles[topic].close()
