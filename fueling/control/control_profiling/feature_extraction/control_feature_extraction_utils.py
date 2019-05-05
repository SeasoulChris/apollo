#!/usr/bin/env python

""" Control feature extraction related utils. """

import os

import colored_glog as glog
import numpy as np

from modules.data.fuel.fueling.control.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils

from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_IDX, MODE_IDX

# Message number in each segment
MSG_PER_SEGMENT = 1000
# Maximum allowed time gap betwee two messages
MAX_PHASE_DELTA = 0.01
# Minimum epsilon value used in compare with zero
MIN_EPSILON = 0.000001


def verify_vehicle_controller(task):
    """Verify if the task has any record file whose controller/vehicle types match config"""
    record_file = next((os.path.join(task, record_file) for record_file in os.listdir(task)
                        if (record_utils.is_record_file(record_file) or
                            record_utils.is_bag_file(record_file))), None)
    if not record_file:
        glog.warn('no valid record file found in task: {}'.format(task))
        return False
    # Read two topics together to avoid looping all messages in the record file twice
    glog.info('verifying vehicle controler in task {}, record {}'.format(task, record_file))
    read_record_func = record_utils.read_record([record_utils.CONTROL_CHANNEL,
                                                 record_utils.HMI_STATUS_CHANNEL])
    messages = read_record_func(record_file)
    glog.info('{} messages for record file {}'.format(len(messages), record_file))
    vehicle_message = get_message_by_topic(messages, record_utils.HMI_STATUS_CHANNEL)
    if not vehicle_message:
        glog.error('no vehicle messages found in task {} record {}'.format(task, record_file))
        return False
    control_message = get_message_by_topic(messages, record_utils.CONTROL_CHANNEL)
    if not control_message:
        glog.error('no control messages found in task {} record {}'.format(task, record_file))
        return False
    return data_matches_config(record_utils.message_to_proto(vehicle_message).current_vehicle,
                               record_utils.message_to_proto(control_message))


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


def extract_data_at_auto_mode(msgs, driving_mode, gear_position):
    """Extract control/chassis data array and filter the control data with selected chassis features"""
    chassis_msgs = collect_message_by_topic(msgs, record_utils.CHASSIS_CHANNEL)
    control_msgs = collect_message_by_topic(msgs, record_utils.CONTROL_CHANNEL)
    chassis_mtx = np.array([extract_chassis_data_from_msg(msg) for msg in chassis_msgs])
    control_mtx = np.array([extract_control_data_from_msg(msg) for msg in control_msgs])
    glog.info('The original chassis msgs size is: {} and original control msgs size is: {}'
              .format(chassis_mtx.shape[0], control_mtx.shape[0]))
    driving_condition = (chassis_mtx[:, MODE_IDX['driving_mode']] == driving_mode)
    gear_condition =  (chassis_mtx[:, MODE_IDX['gear_location']] == gear_position[0])
    for gear_idx in range(1, len(gear_position)):
        gear_condition |= (chassis_mtx[:, MODE_IDX['gear_location']] == gear_position[gear_idx])
    chassis_idx_filtered = np.where(driving_condition & gear_condition)[0]
    chassis_mtx_filtered = np.take(chassis_mtx, chassis_idx_filtered, axis=0)
    control_idx_filtered = []
    chassis_idx_refiltered = []
    chassis_idx = 0
    control_idx = 0
    while (chassis_idx < chassis_mtx_filtered.shape[0]) and (control_idx < control_mtx.shape[0]):
        if (chassis_mtx_filtered[chassis_idx, MODE_IDX['timestamp_sec']] -
            control_mtx[control_idx, FEATURE_IDX['timestamp_sec']]) >= MAX_PHASE_DELTA:
            control_idx += 1
        elif (control_mtx[control_idx, FEATURE_IDX['timestamp_sec']] -
              chassis_mtx_filtered[chassis_idx, MODE_IDX['timestamp_sec']]) >= MAX_PHASE_DELTA:
            chassis_idx += 1
        else:
            control_idx_filtered.append(control_idx)
            chassis_idx_refiltered.append(chassis_idx)
            chassis_idx += 1
            control_idx += 1
    control_mtx_filtered = np.take(control_mtx, control_idx_filtered, axis=0)
    chassis_mtx_refiltered = np.take(chassis_mtx_filtered, chassis_idx_refiltered, axis=0)
    glog.info('The filterd chassis msgs size is: {} and filtered control msgs size is: {}'
              .format(chassis_mtx_refiltered.shape[0], control_mtx_filtered.shape[0]))
    if (chassis_mtx_refiltered.shape[1] > MODE_IDX['throttle_chassis'] and
        chassis_mtx_refiltered.shape[1] > MODE_IDX['brake_chassis']):
        grading_mtx = np.hstack((control_mtx_filtered,
                                 chassis_mtx_refiltered[:, [MODE_IDX['throttle_chassis'],
                                                            MODE_IDX['brake_chassis']]]))
    else:
        grading_mtx = control_mtx_filtered
    return grading_mtx


def get_message_by_topic(messages, topic):
    """Get the first message from list that has specific topic"""
    return next((message for message in messages if message.topic == topic), None)


def collect_message_by_topic(messages, topic):
    """Collect all the messages from list that has specific topic"""
    return [message for message in messages if message.topic == topic]


def get_config_control_profiling():
    """Get configured value in control_profiling_conf.pb.txt"""
    profiling_conf = \
        '/apollo/modules/data/fuel/fueling/control/conf/control_profiling_conf.pb.txt'
    control_profiling = ControlProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, control_profiling)
    return control_profiling


def extract_control_data_from_msg(msg):
    """Extract wanted fields from control message"""
    msg_proto = record_utils.message_to_proto(msg)
    control_latency = msg_proto.latency_stats
    control_header = msg_proto.header
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
            # Features: "State" category
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
            # Features: "Latency" category
            control_latency.total_time_ms,               # 26
            control_latency.total_time_exceeded,         # 27
            # Features" "Time" category
            control_header.timestamp_sec                 # 28
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
            # Features: "State" category
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
            # Features: "Latency" category
            control_latency.total_time_ms,               # 26
            control_latency.total_time_exceeded,         # 27
            # Features" "Time" category
            control_header.timestamp_sec                 # 28
        ])
    return data_array

def extract_chassis_data_from_msg(msg):
    """Extract wanted fields from chassis message"""
    msg_proto = record_utils.message_to_proto(msg)
    chassis_header = msg_proto.header
    if get_config_control_profiling().vehicle_type.find('Mkz') >= 0:
        data_array = np.array([
            # Features: "Status" category
            msg_proto.driving_mode,                          # 0
            msg_proto.gear_location,                         # 1
            # Features: "Time" category
            chassis_header.timestamp_sec,                    # 2
            # Features: "Action" category
            msg_proto.throttle_percentage,                   # 3
            msg_proto.brake_percentage                       # 4
        ])
    else:
        data_array = np.array([
            # Features: "Status" category
            msg_proto.driving_mode,                          # 0
            msg_proto.gear_location,                         # 1
            # Features: "Time" category
            chassis_header.timestamp_sec                     # 2
        ])
    return data_array
