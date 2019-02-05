#!/usr/bin/env python

import errno
import fnmatch
import os

from cyber_py.record import RecordReader, RecordWriter

import fueling.common.s3_utils as s3_utils


def IsRecordFile(path):
    """Naive check if a path is a record."""
    return path.endswith('.record') or fnmatch.fnmatch(path, '*.record.?????')

def ReadRecord(wanted_channels=None):
    """record_path -> [PyBagMessage, ...] or None if error occurs."""
    def ReadRecordFunc(record_path):
        try:
            for msg in RecordReader(s3_utils.AbsPath(record_path)).read_messages():
                if wanted_channels is None or msg.topic in wanted_channels:
                    yield msg
        except Exception as e:
            # Stop poping messages elegantly if exception happends, including the normal StopIteration.
            raise StopIteration
    return ReadRecordFunc

def WriteRecord(path_to_messages):
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
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    writer = RecordWriter(0, 0)
    writer.open(path)
    for msg in py_bag_messages:
        # As a generated record, we ignored the proto desc.
        writer.write_channel(msg.topic, msg.data_type, '')
        writer.write_message(msg.topic, msg.message, msg.timestamp)
    writer.close()
    # Dummy map result.
    return None
