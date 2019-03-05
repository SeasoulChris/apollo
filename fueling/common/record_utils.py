"""Cyber records related utils."""
#!/usr/bin/env python

import fnmatch
import os

from cyber.proto.record_pb2 import Header
from cyber_py.record import RecordReader, RecordWriter
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.routing.proto.routing_pb2 import RoutingResponse

import fueling.common.colored_glog as glog
import fueling.common.file_utils as file_utils

HMI_STATUS_CHANNEL = '/apollo/hmi/status'
CHASSIS_CHANNEL = '/apollo/canbus/chassis'
LOCALIZATION_CHANNEL = '/apollo/localization/pose'
ROUTING_RESPONSE_HISTORY_CHANNEL = '/apollo/routing_response_history'

CHANNEL_TO_TYPE = {
    CHASSIS_CHANNEL: Chassis,
    HMI_STATUS_CHANNEL: HMIStatus,
    LOCALIZATION_CHANNEL: LocalizationEstimate,
    ROUTING_RESPONSE_HISTORY_CHANNEL: RoutingResponse,

}


def is_record_file(path):
    """Naive check if a path is a record."""
    return path.endswith('.record') or fnmatch.fnmatch(path, '*.record.?????')

def read_record(channels, start_time_ns=0, end_time_ns=18446744073709551615):
    """record_path -> [PyBagMessage, ...] or None if error occurs."""
    def read_record_func(record_path):
        """Wrapper function."""
        glog.info('Read record {}'.format(record_path))
        try:
            reader = RecordReader(record_path)
            channel_set = {
                channel
                for channel in set(channels).intersection(reader.get_channellist())
                if reader.get_messagenumber(channel) > 0}
            if channel_set:
                return [msg for msg in reader.read_messages() if (
                    msg.topic in channel_set and
                    msg.timestamp >= start_time_ns and
                    msg.timestamp < end_time_ns)]
        except Exception as err:
            # Stop poping messages elegantly if exception happends, including
            # the normal StopIteration.
            glog.error('Failed to read record {}: {}'.format(record_path, err))
        return []
    return read_record_func

def read_record_header(record_path):
    """record_path -> Header, or None if error occurs."""
    glog.info('Read record header {}'.format(record_path))
    try:
        reader = RecordReader(record_path)
        header = Header()
        header.ParseFromString(reader.get_headerstring())
        if header.message_number > 0:
            return header
        glog.error('No message in record {} or its header is broken'.format(record_path))
    except Exception as e:
        glog.error('Failed to read record header {}: {}'.format(record_path, e))
    return None

def write_record(path_to_messages):
    """
    Write a list of messages to the record path. Note that this is just a Spark
    transformation which needs to be flushed by an action, such as 'count()'.

    PyBagMessage = namedtuple(topic, message, data_type, timestamp)
    """
    # Prepare the input data and output dir.
    path, py_bag_messages = path_to_messages
    file_utils.makedirs(os.path.dirname(path))
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
    return path

def message_to_proto(py_bag_message):
    """Convert an Apollo py_bag_message to proto."""
    proto_type = CHANNEL_TO_TYPE.get(py_bag_message.topic)
    if proto_type is None:
        glog.error('Parser for {} is not implemented!'.format(py_bag_message.topic))
        return None
    proto = proto_type()
    proto.ParseFromString(py_bag_message.message)
    return proto
