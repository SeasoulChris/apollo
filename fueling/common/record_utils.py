"""Cyber records related utils."""
#!/usr/bin/env python

import collections
import fnmatch
import os

import colored_glog as glog

from cyber.proto.record_pb2 import Header
from cyber_py.record import RecordReader, RecordWriter
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.control.proto.control_cmd_pb2 import ControlCommand
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
from modules.drivers.proto.sensor_image_pb2 import CompressedImage
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.routing.proto.routing_pb2 import RoutingResponse

import fueling.common.file_utils as file_utils


CHASSIS_CHANNEL =                  '/apollo/canbus/chassis'
CONTROL_CHANNEL =                  '/apollo/control'
DRIVE_EVENT_CHANNEL =              '/apollo/drive_event'
HMI_STATUS_CHANNEL =               '/apollo/hmi/status'
LOCALIZATION_CHANNEL =             '/apollo/localization/pose'
ROUTING_RESPONSE_HISTORY_CHANNEL = '/apollo/routing_response_history'
FRONT_12mm_CHANNEL =               '/apollo/sensor/camera/front_12mm/image/compressed'
FRONT_6mm_CHANNEL =                '/apollo/sensor/camera/front_6mm/image/compressed'
LEFT_FISHEYE_CHANNEL =             '/apollo/sensor/camera/left_fisheye/image/compressed'
REAR_6mm_CHANNEL =                 '/apollo/sensor/camera/rear_6mm/image/compressed'
RIGHT_FISHEYE_CHANNEL =            '/apollo/sensor/camera/right_fisheye/image/compressed'

CHANNEL_TO_TYPE = {
    CHASSIS_CHANNEL: Chassis,
    CONTROL_CHANNEL: ControlCommand,
    HMI_STATUS_CHANNEL: HMIStatus,
    LOCALIZATION_CHANNEL: LocalizationEstimate,
    ROUTING_RESPONSE_HISTORY_CHANNEL: RoutingResponse,
    FRONT_6mm_CHANNEL: CompressedImage,
    FRONT_12mm_CHANNEL: CompressedImage,
    REAR_6mm_CHANNEL: CompressedImage,
    LEFT_FISHEYE_CHANNEL: CompressedImage,
    RIGHT_FISHEYE_CHANNEL: CompressedImage,
}


def is_record_file(path):
    """Naive check if a path is a record."""
    return path.endswith('.record') or fnmatch.fnmatch(path, '*.record.?????')

def read_record(channels=None, start_time_ns=0, end_time_ns=18446744073709551615):
    """record_path -> [PyBagMessage, ...] or [] if error occurs."""
    def read_record_func(record_path):
        """Wrapper function."""
        glog.info('Read record {}'.format(record_path))
        try:
            reader = RecordReader(record_path)
            channel_set = {channel for channel in reader.get_channellist()
                           if reader.get_messagenumber(channel) > 0}
            if channels:
                channel_set.intersection_update(channels)
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

def messages_to_proto_dict(sort_by_msg_time=True, sort_by_header_time=False):
    """py_bag_messages -> {topic:[protos]]."""
    def converter_func(py_bag_messages):
        """Wrapper function."""
        if sort_by_msg_time:
            py_bag_messages = sorted(py_bag_messages, key=lambda msg: msg.timestamp)

        result = collections.defaultdict(list)
        for msg in py_bag_messages:
            result[msg.topic].append(message_to_proto(msg))

        if sort_by_header_time:
            for topic, protos in result.iteritems():
                result[topic] = sorted(protos, key=lambda proto: proto.header.timestamp_sec)
        return result
    return converter_func

def get_map_name_from_records(records_dir):
    """Get the map_name from a records_dir by /apollo/hmi/status channel"""
    map_list = os.listdir('/apollo/modules/map/data/')
    # get the map_dict mapping follow the hmi Titlecase. E.g.: "Hello World" -> "hello_world".
    map_dict = {map_name.replace('_', ' ').title(): map_name for map_name in map_list}
    reader = read_record([HMI_STATUS_CHANNEL])
    glog.info('Try getting map name from {}'.format(records_dir))
    records = [os.path.join(records_dir, filename) for filename in os.listdir(records_dir)
                                                   if is_record_file(filename)]
    for record in records:
        for msg in reader(record):
            hmi_status = message_to_proto(msg)
            map_name = map_dict[str(hmi_status.current_map)]
            glog.info('Get map name "{}" from record {}'.format(map_name, record))
            return map_name
    glog.error('Failed to get map_name')
