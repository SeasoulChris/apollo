#!/usr/bin/env python

""" Vehicle dynamics feature extraction related utils. """

import os

import colored_glog as glog
import numpy as np

from modules.data.fuel.fueling.profiling.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, MODE_IDX, POSE_IDX


def extract_data_two_channels(msgs, driving_mode, gear_position):
    """Extract control/chassis data array and filter the control data with selected chassis features"""
    chassis_msgs = collect_message_by_topic(msgs, record_utils.CHASSIS_CHANNEL)
    control_msgs = collect_message_by_topic(msgs, record_utils.CONTROL_CHANNEL)

    chassis_mtx = np.array([extract_chassis_data_from_msg(msg)
                            for msg in chassis_msgs])
    control_mtx = np.array([extract_control_data_from_msg(msg)
                            for msg in control_msgs])

    glog.info('The original msgs size are: chassis {}, control {}'
              .format(chassis_mtx.shape[0], control_mtx.shape[0]))
    if (chassis_mtx.shape[0] == 0 or control_mtx.shape[0] == 0):
        return np.take(control_mtx, [], axis=0)
    # First, filter the chassis data with desired driving modes and gear locations
    driving_condition = (
        chassis_mtx[:, MODE_IDX['driving_mode']] == driving_mode)
    gear_condition = (
        chassis_mtx[:, MODE_IDX['gear_location']] == gear_position[0])
    for gear_idx in range(1, len(gear_position)):
        gear_condition |= (
            chassis_mtx[:, MODE_IDX['gear_location']] == gear_position[gear_idx])
    chassis_idx_filtered = np.where(driving_condition & gear_condition)[0]
    chassis_mtx_filtered = np.take(chassis_mtx, chassis_idx_filtered, axis=0)
    # Second, filter the control data with existing chassis and localization sequence_num
    control_idx_by_chassis = np.in1d(control_mtx[:, FEATURE_IDX['chassis_sequence_num']],
                                     chassis_mtx_filtered[:, MODE_IDX['sequence_num']])
    control_mtx_rtn = control_mtx[control_idx_by_chassis, :]
    # Third, delete the control data with inverted-sequence chassis and localization sequence_num
    # (in very rare cases, the sequence number in control record is like ... 100, 102, 101, 103 ...)
    inv_seq_chassis = (
        np.diff(control_mtx_rtn[:, FEATURE_IDX['chassis_sequence_num']]) < 0)
    control_idx_inv_seq = np.where(np.insert(inv_seq_chassis, 0, 0))
    control_mtx_rtn = np.delete(control_mtx_rtn, control_idx_inv_seq, axis=0)
    # Fourth, filter the chassis and localization data with filtered control data
    chassis_idx_rtn = []
    chassis_idx = 0
    for control_idx in range(control_mtx_rtn.shape[0]):
        while (control_mtx_rtn[control_idx, FEATURE_IDX['chassis_sequence_num']] !=
               chassis_mtx_filtered[chassis_idx, MODE_IDX['sequence_num']]):
            chassis_idx += 1
        chassis_idx_rtn.append(chassis_idx)
    chassis_mtx_rtn = np.take(chassis_mtx_filtered, chassis_idx_rtn, axis=0)
    glog.info('The filtered msgs size are: chassis {}, control {}'
              .format(chassis_mtx_rtn.shape[0], control_mtx_rtn.shape[0]))
    # Finally, rebuild the grading mtx with the control data combined with chassis and localizaiton data
    # TODO(fengzongbao) Filter acceleration_reference by positive and negative to throttle and brake
    if (control_mtx_rtn.shape[0] > 0):
        # First, merge the chassis data into control data matrix
        if (chassis_mtx_rtn.shape[1] > MODE_IDX['brake_chassis']):
            grading_mtx = np.hstack((control_mtx_rtn,
                                     chassis_mtx_rtn[:, [MODE_IDX['throttle_chassis'],
                                                         MODE_IDX['brake_chassis']]]))
        else:
            grading_mtx = np.hstack((control_mtx_rtn,
                                     np.zeros((control_mtx_rtn.shape[0], 2))))
    else:
        grading_mtx = control_mtx_rtn
    return grading_mtx


def get_message_by_topic(messages, topic):
    """Get the first message from list that has specific topic"""
    return next((message for message in messages if message.topic == topic), None)


def collect_message_by_topic(messages, topic):
    """Collect all the messages from list that has specific topic"""
    return [message for message in messages if message.topic == topic]


def get_profiling_config():
    """Get configured value in control_profiling_conf.pb.txt"""
    profiling_conf = \
        '/apollo/modules/data/fuel/fueling/profiling/conf/vehicle_dynamics_profiling_conf.pb.txt'
    control_profiling = ControlProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, control_profiling)
    return control_profiling


def extract_control_data_from_msg(msg):
    """Extract wanted fields from control message"""
    msg_proto = record_utils.message_to_proto(msg)
    control_header = msg_proto.header
    input_debug = msg_proto.debug.input_debug
    if get_profiling_config().controller_type == 'Lon_Lat_Controller':
        control_lon = msg_proto.debug.simple_lon_debug
        control_lat = msg_proto.debug.simple_lat_debug
        data_array = np.array([
            # Features: "Command" category
            msg_proto.acceleration,                         # 0
            msg_proto.steering_target,                      # 1
            # Features: "Reference" category
            control_lon.current_acceleration,               # 2
            control_lat.steering_position,                  # 3
            # Features: "Header" category
            control_header.timestamp_sec,                   # 4
            control_header.sequence_num,                    # 5
            # Features: "Input Info" category
            input_debug.canbus_header.timestamp_sec,        # 6
            input_debug.canbus_header.sequence_num,         # 7
        ])
    else:
        control_mpc = msg_proto.debug.simple_mpc_debug
        data_array = np.array([
            # Features: "Command" category
            msg_proto.acceleration,                         # 0
            msg_proto.steering_target,                      # 1
            # Features: "State" category
            control_mpc.acceleration_feedback,              # 2
            control_mpc.steering_position,                  # 3
            # Features: "Header" category
            control_header.timestamp_sec,                   # 4
            control_header.sequence_num,                    # 5
            # Features: "Input Info" category
            input_debug.canbus_header.timestamp_sec,        # 6
            input_debug.canbus_header.sequence_num          # 7
        ])

    return data_array


def extract_chassis_data_from_msg(msg):
    """Extract wanted fields from chassis message"""
    msg_proto = record_utils.message_to_proto(msg)
    chassis_header = msg_proto.header
    if get_profiling_config().vehicle_type.find('Mkz') >= 0:
        data_array = np.array([
            # Features: "Status" category
            msg_proto.driving_mode,                          # 0
            msg_proto.gear_location,                         # 1
            # Features: "Header" category
            chassis_header.timestamp_sec,                    # 2
            chassis_header.sequence_num,                     # 3
            # Features: "Action" category
            msg_proto.throttle_percentage,                   # 4
            msg_proto.brake_percentage                       # 5
        ])
    else:
        data_array = np.array([
            # Features: "Status" category
            msg_proto.driving_mode,                          # 0
            msg_proto.gear_location,                         # 1
            # Features: "Header" category
            chassis_header.timestamp_sec,                    # 2
            chassis_header.sequence_num                      # 3
        ])
    return data_array
