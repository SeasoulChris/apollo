#!/usr/bin/env python

"""Utility functions for jobs that uses streaming"""

from collections import namedtuple
import operator
import os
import pickle
import re
import shutil
import stat
import time

import colored_glog as glog

import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils

STREAMING_PATH = 'modules/streaming'
STREAMING_RECORDS = 'records'
STREAMING_DATA = 'data'
STREAMING_IMAGE = 'images'
STREAMING_CONF = 'conf'

# utility functions
def get_todo_records(root_dir, target_dir=None):
    """
    Get records that have completed the serialization by streaming job, and not have been processed
    ---
    Parameters:
    1. root_dir, for example '/mnt/bos' for production, and '/apollo' for test
    2. target_dir, the directory where results are generated, it's purely a setting that
    the application defines by itself.  For example 'modules/perception/whatever/generated'.
    After processing successfully we would want to put a COMPLETE file under the task to avoid it
    being processed again.
    ---
    Returns:
    1. A list of record files with absolute paths, and
    2. A list of tasks/dirs under which every single record has completed the serialization
    """
    todo_records = []
    todo_tasks = []
    records_path = get_streaming_records(root_dir)
    for task_file in os.listdir(records_path):
        if target_dir is not None:
            target_path = os.path.join(os.path.join(root_dir, target_dir), task_file)
            if os.path.exists(os.path.join(target_path, 'COMPLETE')):
                continue
        task_file_path = os.path.join(records_path, task_file)
        with open(task_file_path) as read_task_file:
            records = list(read_task_file.readlines())
            if not records:
                continue
            for record in records:
                record = record.strip()
                if not record_utils.is_record_file(record):
                    continue
                module_target = os.path.dirname(locate_target_record(root_dir, target_dir, record))
                if os.path.exists(os.path.join(module_target, 'COMPLETE')):
                    continue
                if is_serialization_completed(root_dir, record):
                    todo_records.append(record)
                    if task_file_path not in todo_tasks:
                        todo_tasks.append(task_file_path)
    return todo_records, todo_tasks

def load_meta_data(root_dir, record_file, topics):
    """
    Load meta data which contains key arguments that the application needs, with
    specified topics
    It will be like a list of the following named tuples:
    (topic, timestamp, {field_name_1: field_value_1, field_name_2: field_value_2, ...}, objpath),
    in which the interested fields can be defined in streaming config file
    """
    tuples_list = []
    for topic in topics:
        lines = load_topic(root_dir, record_file, topic)
        for line in lines:
            tuples_list.append(build_meta_from_line(root_dir, record_file, topic, line))
    return tuples_list

def load_topic(root_dir, record_file, topic):
    """Load specific topic.  Basically for checking if topic is existing in some record file"""
    data_dir = record_to_stream_path(record_file, root_dir, STREAMING_DATA)
    topic_file_path = os.path.join(data_dir, topic_to_file_name(topic))
    if not os.path.exists(topic_file_path):
        return []
    with open(topic_file_path) as read_topic_file:
        return read_topic_file.readlines()

def load_messages(root_dir, record_file, topics):
    """
    Load multiple topics, merge them together sorting by timestamp.
    Message objs' path are included
    """
    return sorted(load_meta_data(root_dir, record_file, topics), key=operator.itemgetter(1))

def load_message_obj(message_file_path):
    """Load the message objects that stored previously in file system"""
    # Do not load the object if it does not exist or if it is compressed camera message
    if not os.path.exists(message_file_path) or message_file_path.find('compressed') != -1:
        return None
    with open(message_file_path, 'rb') as message_file:
        return pickle.load(message_file)
    return None

def load_image_data(image_file_path):
    """Explicitly load image data"""
    if not os.path.exists(image_file_path) or image_file_path.find('compressed') == -1:
        return None
    with open(image_file_path, 'rb') as image_file:
        return image_file.read()
    return None

# helper functions
def record_to_stream_path(record_path, root_dir, sub_folder_name):
    """Convert absolute path to the corresponding stream path"""
    sub_folder = os.path.join(STREAMING_PATH, sub_folder_name)
    if record_path.startswith(root_dir):
        relative_path = record_path.replace(root_dir+'/', '', 1).strip()
    else:
        relative_path = record_path.strip('/ \n')
    return os.path.join(os.path.join(root_dir, sub_folder), relative_path)

def stream_to_record_path(stream_path, root_dir, sub_folder_name):
    """Convert stream path to the corresponding record path"""
    sub_folder = os.path.join(STREAMING_PATH, sub_folder_name)
    if stream_path.startswith(root_dir):
        relative_path = stream_path.replace(root_dir+'/', '', 1).replace(sub_folder+'/', '', 1)
        return os.path.join(root_dir, relative_path)
    relative_path = stream_path.strip('/ \n').replace(sub_folder+'/', '', 1)
    return os.path.join('/', relative_path)

def list_records_for_task(task_path):
    """List all records for a specific task"""
    with open(task_path) as read_task_file:
        records = read_task_file.readlines()
    return [record.strip() for record in records]

def target_partition_to_records(root_dir, target_partition, slice_size):
    """Revert a partition to original record file paths"""
    seq = int(re.search(r'[\S]*\d{14}#SS([\d]*)$', target_partition, re.M|re.I).group(1))
    task_id = os.path.basename(os.path.dirname(target_partition))
    src_records = list_records_for_task(
        os.path.join(get_streaming_records(root_dir), task_id))
    seq_range = ['{:05d}'.format(rec) for rec in range(seq * slice_size, (seq+1) * slice_size)]
    for ret_record in src_records:
        ret_record = ret_record.strip()
        if record_utils.is_record_file(ret_record) and ret_record.rsplit('.', 1)[1] in seq_range:
            yield ret_record

def is_serialization_completed(root_dir, record_file):
    """Check if serialization is done for the specified record file"""
    record_dir = record_to_stream_path(record_file, root_dir, STREAMING_DATA)
    return os.path.exists(os.path.join(record_dir, 'COMPLETE'))

def get_streaming_records(root_dir):
    """Get streaming record path which the streaming process monitors"""
    return os.path.join(root_dir, os.path.join(STREAMING_PATH, STREAMING_RECORDS))

def get_message_id(timestamp, topic):
    """
    Form a message ID by using timestamp and topic name, 
    which combined together should be able to represent an unique message
    """
    return '{}{}'.format(timestamp, topic_to_file_name(topic))

def topic_to_file_name(topic):
    """Convert / in topic to - to make it compatible with file system"""
    return topic.replace('/', '_')

def file_name_to_topic(topic):
    """Convert _ in file name back to topic"""
    return topic.replace('_', '/')

def write_to_file(file_path, mode, message):
    """Write message to file, create new one if not existing"""
    with open(file_path, mode) as file_to_write:
        file_to_write.write(str(message))
    file_stat = os.stat(file_path)
    os.chmod(file_path, file_stat.st_mode | stat.S_IWOTH)

def write_message_obj(record_dir, renamed_topic, py_message, header_time):
    """Write message object in binary to file system"""
    message_file_name = get_message_id(header_time, renamed_topic)
    with open(os.path.join(record_dir, message_file_name), 'wb') as message_file:
        # For compressed camera messages, write binary data instead of message objects
        if renamed_topic.endswith('compressed'):
            proto = record_utils.message_to_proto(py_message)
            # pickle.dump(proto.data, message_file, pickle.HIGHEST_PROTOCOL)
            message_file.write(proto.data)
        else:
            pickle.dump(py_message.message, message_file, pickle.HIGHEST_PROTOCOL)

def build_meta_from_line(root_dir, record_file, topic, line):
    """Parse the line in topic file and form a tuple"""
    timestamp = None
    fields = None
    message_meta = namedtuple('MessageMeta', ['topic', 'timestamp', 'fields', 'objpath'])
    reg_search = re.search(r'^(\d*)', line.strip(), re.M|re.I)
    if reg_search is not None:
        timestamp = reg_search.group(1)
    reg_search = re.search(r'^\d*,(\{.+\})$', line.strip(), re.M|re.I)
    if reg_search is not None:
        fields = reg_search.group(1)
    data_dir = record_to_stream_path(record_file, root_dir, STREAMING_DATA)
    objpath = os.path.join(data_dir, get_message_id(timestamp, topic))
    return message_meta(topic=topic, timestamp=timestamp, fields=fields, objpath=objpath)

def retry(func, params, retry_times):
    """A wrapper to retry calling given function for specified times"""
    while retry_times > 0:
        try:
            ret = func(*params)
            return ret
        except Exception as expn:
            error_msg = 'func failed with error {}. params: {}'.format(expn, params)
            glog.error(error_msg)
            retry_times -= 1
            if retry_times <= 0:
                raise Exception(error_msg)
            time.sleep(1)

def locate_target_record(root_dir, module_dir, record):
    """Determine the target dir of records in particular modules"""
    if record.find(root_dir) < 0:
        # It's possible that original record path does not start with root_dir path,
        # in this case just append the original record to module dir
        dst_record = os.path.join(root_dir, module_dir, record[1:])
    else:
        # Otherwise, replace the exact next sub dir of root_dir and with module_dir
        dst_record = '{}{}'.format(os.path.join(root_dir, module_dir),
                                   record[record.find('/', len(root_dir) + 1) : ])
    return dst_record

