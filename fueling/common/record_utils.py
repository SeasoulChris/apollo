#!/usr/bin/env python

import fnmatch

from cyber_py.record import RecordReader, RecordWriter

import fueling.common.s3_utils as s3_utils


def IsRecordFile(path):
    """Naive check if a path is a record."""
    return path.endswith('.record') or fnmatch.fnmatch(path, '*.record.?????')

def ReadRecord(record_key):
    """record_path -> [PyBagMessage, ...] or None if error occurs."""
    try:
        for msg in RecordReader(s3_utils.KeyToMountPath(record_key)).read_messages():
            yield msg
    except Exception as e:
        # Stop poping messages elegantly if exception happends, including the normal StopIteration.
        raise StopIteration

def WriteRecord(record_key, py_bag_messages, is_sorted=False,
                file_segmentation_size_kb=0, file_segmentation_interval_sec=60):
    """
    Write a list of messages to the record path. Note that this is just a Spark
    transformation which needs to be flushed by an action, such as 'count()'.

    PyBagMessage = namedtuple(topic, message, data_type, timestamp')
    """
    writer = RecordWriter(file_segmentation_size_kb, file_segmentation_interval_sec)
    writer.open(s3_utils.KeyToMountPath(record_key))

    msgs = py_bag_messages if is_sorted else sorted(py_bag_messages, key=lambda msg: msg.timestamp)
    for msg in msgs:
        writer.WriteMessage(msg.topic, msg.message, msg.timestamp, msg.data_type)
    writer.close()

    # Dummy map result.
    return None
