#!/usr/bin/env python

import fnmatch

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
    path, py_bag_messages = path_to_messages

    writer = RecordWriter(0, 0)
    writer.open(s3_utils.AbsPath(path))
    for msg in py_bag_messages:
        writer.WriteMessage(msg.topic, msg.message, msg.timestamp, msg.data_type)
    writer.close()
    # Dummy map result.
    return None
