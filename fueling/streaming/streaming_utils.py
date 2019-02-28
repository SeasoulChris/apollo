#!/usr/bin/env python

"""Utility functions for jobs that uses streaming"""

import os
import re

STREAMING_PATH = 'modules/data/fuel/testdata/streaming'
STREAMING_RECORDS = 'records'
STREAMING_DATA = 'data'

# utility functions
def get_todo_records(root_dir, target_dir=None):
    """
    Get records that have completed the serialization by streaming job, and not have been processed
    ---
    Parameters:
    1. root_dir, for example '/mnt/bos' for production, and '/apollo' for test
    2. target_dir, the directory where results are generated, it's purely a setting that the application
    defines by itself.  For example 'modules/perception/whatever/generated'.
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
        todo_tasks.append(task_file_path)
        with open(task_file_path) as read_task_file:
            records = read_task_file.readlines()
            for record in records:
                if is_serialization_completed(root_dir, record.strip()):
                    todo_records.append(record)
                elif task_file_path in todo_tasks:
                    todo_tasks.remove(task_file_path)
    return todo_records, todo_tasks

def load_meta_data(record_file, topics):
    """
    Load meta data which contains key arguments that the application needs, with
    specified topics
    It will be like a list of the following tuples:
    (topic, timestamp, {field_name_1: field_value_1, field_name_2: field_value_2, ...}), 
    in which the interested fields can be defined in streaming config file
    """
    tuples_list = []
    root_dir = record_file[:record_file.find(STREAMING_PATH)-1]
    for topic in topics:
        topic_file_path = os.path.join(record_to_stream_path(root_dir, record_file), \
            topic_to_file_name(topic))
        with open(topic_file_path) as read_topic_file:
            lines = read_topic_file.readlines()
            for line in lines:
                tuples_list.append(build_tuple_from_line(topic, line))
    return tuples_list

# helper functions
def record_to_stream_path(record_path, root_dir):
    """Convert absolute path to the corresponding stream path"""
    relative_path = record_path.replace(root_dir+'/', '', 1)
    return os.path.join(os.path.join(root_dir, STREAMING_PATH), relative_path)

def list_records_for_task(task_path):
    """List all records for a specific task"""
    with open(task_path) as read_task_file:
        records = read_task_file.readlines()
    return list(records)

def stream_to_record_path(stream_path, root_dir):
    """Convert stream path to the corresponding record path"""
    relative_path = stream_path.replace(root_dir, '', 1).replace(STREAMING_PATH, '', 1)
    return os.path.join(root_dir, relative_path)

def is_serialization_completed(root_dir, record_file):
    """Check if serialization is done for the specified record file"""
    stream_path = record_to_stream_path(record_file, root_dir)
    return os.path.exists(os.path.join(stream_path, 'COMPLETE'))

def get_streaming_records(root_dir):
    """Get streaming record path which the streaming process monitors"""
    return os.path.join(root_dir, os.path.join(STREAMING_PATH, STREAMING_RECORDS))

def topic_to_file_name(topic):
    """Convert / in topic to - to make it compatible with file system"""
    return topic.replace('/', '_')

def build_tuple_from_line(topic, line):
    """Parse the line in topic file and form a tuple"""
    timestamp = None
    fields = None
    reg_search = re.search(r'^(\d*)', line.strip(),  re.M|re.I)
    if reg_search is not None:
        timestamp = reg_search.group(1)
    reg_search = re.search(r'^\d*, (\{.+\})$', line.strip(),  re.M|re.I)
    if reg_search is not None:
        fields = reg_search.group(1)
    return (topic, timestamp, fields)