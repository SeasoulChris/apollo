"""Cyber records related utils."""
#!/usr/bin/env python

import collections
import fnmatch
import os

import colored_glog as glog

from cyber.proto.record_pb2 import Header
from cyber_py.record import RecordReader
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.control.proto.control_cmd_pb2 import ControlCommand
from modules.dreamview.proto.hmi_status_pb2 import HMIStatus
from modules.drivers.proto.sensor_image_pb2 import CompressedImage
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.routing.proto.routing_pb2 import RoutingResponse

from fueling.common.mongo_utils import Mongo
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta


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

def message_to_proto(py_bag_message):
    """Convert an Apollo py_bag_message to proto."""
    proto_type = CHANNEL_TO_TYPE.get(py_bag_message.topic)
    if proto_type is None:
        glog.error('Parser for {} is not implemented!'.format(py_bag_message.topic))
        return None
    proto = proto_type()
    try:
        proto.ParseFromString(py_bag_message.message)
    except Exception as e:
        glog.error('Failed to parse message from {}: {}'.format(py_bag_message.topic, e))
        return None
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

def lookup_map_for_dirs(record_dirs, default_map='Unknown', collection='records'):
    """
    [record_dir] -> [(record_dir, map_name)], by looking up the MongoDB.
    This util only works in cluster.
    """
    dirs = list(record_dirs)
    lookup_condition = {'dir': {'$in': dirs}}
    lookup_field = {'dir': 1, 'hmi_status.current_map': 1}
    docs = Mongo.collection(collection).find(lookup_condition, lookup_field)

    dir_to_map = {}
    for doc in docs:
        # Map name already found.
        if doc['dir'] in dir_to_map:
            continue
        record_meta = Mongo.doc_to_pb(doc, RecordMeta())
        map_name = record_meta.hmi_status.current_map
        if map_name:
            glog.info('Found map "{}" for task {}'.format(map_name, record_meta.dir))
            dir_to_map[record_meta.dir] = map_name
    return [(record_dir, dir_to_map.get(record_dir, default_map)) for record_dir in dirs]

def get_vehicle_id_from_records(records, default_id='Unknown'):
    """Get vehicle ID from records."""
    reader = read_record([HMI_STATUS_CHANNEL])
    for record in records:
        for msg in reader(record):
            hmi_status = message_to_proto(msg)
            if hmi_status is None:
                continue
            vehicle_id = hmi_status.current_vehicle
            if vehicle_id:
                glog.info('Got vehicle ID "{}" from record {}'.format(vehicle_id, record))
                return vehicle_id
    glog.error('Failed to get vehicle ID, fallback to: {}'.format(default_id))
    return default_id

def lookup_vehicle_for_dirs(record_dirs, default_vehicle='Unknown', collection='records'):
    """
    [record_dir] -> [(record_dir, vehicle_id)], by looking up the MongoDB.
    This util only works in cluster.
    """
    dirs = list(record_dirs)
    lookup_condition = {'dir': {'$in': dirs}}
    lookup_field = {'dir': 1, 'hmi_status.current_vehicle': 1}
    docs = Mongo.collection(collection).find(lookup_condition, lookup_field)

    dir_to_vehicle = {}
    for doc in docs:
        # Vehicle ID already found.
        if doc['dir'] in dir_to_vehicle:
            continue
        record_meta = Mongo.doc_to_pb(doc, RecordMeta())
        vehicle = record_meta.hmi_status.current_vehicle
        if vehicle:
            glog.info('Found vehicle "{}" for task {}'.format(vehicle, record_meta.dir))
            dir_to_vehicle[record_meta.dir] = vehicle
    return [(record_dir, dir_to_vehicle.get(record_dir, default_vehicle)) for record_dir in dirs]
