"""Cyber records related utils."""
#!/usr/bin/env python

import errno
import fnmatch
import os

import glog

from cyber_py.record import RecordReader, RecordWriter

import fueling.common.s3_utils as s3_utils


def is_record_file(path):
    """Naive check if a path is a record."""
    return path.endswith('.record') or fnmatch.fnmatch(path, '*.record.?????')

def read_record(wanted_channels=None):
    """record_path -> [PyBagMessage, ...] or None if error occurs."""
    def read_record_func(record_path):
        """Wrapper function."""
        try:
            for msg in RecordReader(s3_utils.AbsPath(record_path)).read_messages():
                if wanted_channels is None or msg.topic in wanted_channels:
                    yield msg
        except Exception:
            # Stop poping messages elegantly if exception happends, including
            # the normal StopIteration.
            raise StopIteration
    return read_record_func

def write_record(path_to_messages):
    """
    Write a list of messages to the record path. Note that this is just a Spark
    transformation which needs to be flushed by an action, such as 'count()'.

    PyBagMessage = namedtuple(topic, message, data_type, timestamp)
    """
    # Prepare the input data and output dir.
    path, py_bag_messages = path_to_messages
    path = s3_utils.AbsPath(path)
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
    glog.info('Write record {}'.format(path))
    writer = RecordWriter(0, 0)
    writer.open(path)
    topics = set()
    for msg in py_bag_messages:
        if msg.topic not in topics:
            # As a generated record, we ignored the proto desc.
            writer.write_channel(msg.topic, msg.data_type, '')
            topics.add(msg.topic)
        writer.write_message(msg.topic, msg.message, msg.timestamp)
    writer.close()
    # Dummy map result.
    return None
