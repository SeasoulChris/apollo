#!/usr/bin/env python

""" Control feature extraction related utils. """

import os

import colored_glog as glog
import numpy as np

import common.proto_utils as proto_utils

from modules.data.fuel.fueling.control.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.record_utils as record_utils


def verify_vehicle_controller(task):
    """Verify if the task has any record file whose controller/vehicle types match config"""
    record_file = next((os.path.join(task, record_file) for record_file in os.listdir(task)
                        if record_utils.is_record_file(record_file)), None)
    if not record_file:
        glog.warn('no valid record file found in task: {}'.format(task))
        return False
    # Read two topics together to avoid looping all messages in the record file twice
    read_record_func = record_utils.read_record([record_utils.CONTROL_CHANNEL,
                                                 record_utils.HMI_STATUS_CHANNEL])
    messages = read_record_func(record_file)
    glog.info('{} messages for record file {}'.format(len(messages), record_file))
    vehicle_type = record_utils.message_to_proto(
        get_message_by_topic(messages, record_utils.HMI_STATUS_CHANNEL)).current_vehicle
    controller_type = record_utils.message_to_proto(
        get_message_by_topic(messages, record_utils.CONTROL_CHANNEL))
    return data_matches_config(vehicle_type, controller_type)

def data_matches_config(vehicle_type, controller_type):
    """Compare the data retrieved in record file and configured value and see if matches"""
    conf_vehicle_type = get_config_control_profiling().vehicle_type
    conf_controller_type = get_config_control_profiling().controller_type
    if conf_vehicle_type != vehicle_type:
        glog.warn('mismatch between record vehicle {} and configed {}'
                  .format(vehicle_type, conf_vehicle_type))
        return False
    if controller_type.debug.simple_lat_debug and controller_type.debug.simple_lon_debug:
        if conf_controller_type != 'Lon_Lat_Controller':
            glog.warn('mismatch between record controller Lon_Lat_Controller and configed {}'
                      .format(conf_controller_type))
            return False
    elif controller_type.debug.simple_mpc_debug:
        if conf_controller_type != 'Mpc_Controller':
            glog.warn('mismatch between record controller Mpc_Controller and configed {}'
                      .format(conf_controller_type))
            return False
    else:
        glog.warn('no controller type found in records')
        return False
    return True

def get_message_by_topic(messages, topic):
    """Get the first message from list that has specific topic"""
    return next((message for message in messages if message.topic == topic), None)

def get_config_control_profiling():
    """Get configured value in control_profiling_conf.pb.txt"""
    profiling_conf = \
        '/apollo/modules/data/fuel/fueling/control/conf/control_profiling_conf.pb.txt'
    control_profiling = ControlProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, control_profiling)
    return control_profiling

def extract_data_from_msg(msg):
    """Extract wanted fields from control message"""
    msg_proto = record_utils.message_to_proto(msg)
    if get_config_control_profiling().controller_type == 'Lon_Lat_Controller':
        control_lon = msg_proto.debug.simple_lon_debug
        control_lat = msg_proto.debug.simple_lat_debug
        data_array = np.array([
            # Features: "Refernce" category
            control_lon.station_reference,               # 0
            control_lon.speed_reference,                 # 1
            control_lon.acceleration_reference,          # 2
            control_lat.ref_heading,                     # 3
            control_lat.ref_heading_rate,                # 4
            control_lat.curvature,                       # 5
            # Features: "Error" category
            control_lon.station_error,                   # 6
            control_lon.speed_error,                     # 7
            control_lat.lateral_error,                   # 8
            control_lat.lateral_error_rate,              # 9
            control_lat.heading_error,                   # 10
            control_lat.heading_error_rate,              # 11
            # Features: "Command" category
            msg_proto.throttle,                          # 12
            msg_proto.brake,                             # 13
            msg_proto.acceleration,                      # 14
            msg_proto.steering_target,                   # 15
            # Features: "Status" category
            control_lon.current_station,                 # 16
            control_lon.current_speed,                   # 17
            control_lon.current_acceleration,            # 18
            control_lon.current_jerk,                    # 19
            control_lat.lateral_acceleration,            # 20
            control_lat.lateral_jerk,                    # 21
            control_lat.heading,                         # 22
            control_lat.heading_rate,                    # 23
            control_lat.heading_acceleration,            # 24
            control_lat.heading_jerk,                    # 25
        ])
    else:
        control_mpc = msg_proto.debug.simple_mpc_debug
        data_array = np.array([
            # Features: "Refernce" category
            control_mpc.station_reference,               # 0
            control_mpc.speed_reference,                 # 1
            control_mpc.acceleration_reference,          # 2
            control_mpc.ref_heading,                     # 3
            control_mpc.ref_heading_rate,                # 4
            control_mpc.curvature,                       # 5
            # Features: "Error" category
            control_mpc.station_error,                   # 6
            control_mpc.speed_error,                     # 7
            control_mpc.lateral_error,                   # 8
            control_mpc.lateral_error_rate,              # 9
            control_mpc.heading_error,                   # 10
            control_mpc.heading_error_rate,              # 11
            # Features: "Command" category
            msg_proto.throttle,                          # 12
            msg_proto.brake,                             # 13
            msg_proto.acceleration,                      # 14
            msg_proto.steering_target,                   # 15
            # Features: "Status" category
            control_mpc.station_feedback,                # 16
            control_mpc.speed_feedback,                  # 17
            control_mpc.acceleration_feedback,           # 18
            control_mpc.jerk_feedback,                   # 19
            control_mpc.lateral_acceleration,            # 20
            control_mpc.lateral_jerk,                    # 21
            control_mpc.heading,                         # 22
            control_mpc.heading_rate,                    # 23
            control_mpc.heading_acceleration,            # 24
            control_mpc.heading_jerk,                    # 25
        ])
    return data_array
