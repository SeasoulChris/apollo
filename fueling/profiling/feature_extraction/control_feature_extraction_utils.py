#!/usr/bin/env python

""" Control feature extraction related utils. """

import os

from absl import logging
import numpy as np

from modules.data.fuel.fueling.profiling.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, MODE_IDX, POSE_IDX

# Message number in each segment
MSG_PER_SEGMENT = 3000
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
        logging.warning('no valid record file found in task: {}'.format(task))
        return False
    # Read two topics together to avoid looping all messages in the record file twice
    logging.info('verifying vehicle controler in task {}, record {}'.format(task, record_file))
    read_record_func = record_utils.read_record([record_utils.CONTROL_CHANNEL,
                                                 record_utils.HMI_STATUS_CHANNEL])
    messages = read_record_func(record_file)
    logging.info('{} messages for record file {}'.format(
        len(messages), record_file))
    vehicle_message = get_message_by_topic(
        messages, record_utils.HMI_STATUS_CHANNEL)
    if not vehicle_message:
        logging.error('no vehicle messages found in task {} record {}'.format(task, record_file))
        return False
    control_message = get_message_by_topic(messages, record_utils.CONTROL_CHANNEL)
    if not control_message:
        logging.error('no control messages found in task {} record {}'.format(task, record_file))
        return False
    return data_matches_config(record_utils.message_to_proto(vehicle_message).current_vehicle,
                               record_utils.message_to_proto(control_message))


def data_matches_config(vehicle_type, controller_type):
    """Compare the data retrieved in record file and configured value and see if matches"""
    conf_vehicle_type = get_config_control_profiling().vehicle_type
    conf_controller_type = get_config_control_profiling().controller_type
    if conf_vehicle_type != vehicle_type:
        logging.warning('mismatch between record vehicle {} and configed {}'
                  .format(vehicle_type, conf_vehicle_type))
        return False
    if controller_type.debug.simple_lat_debug and controller_type.debug.simple_lon_debug:
        if conf_controller_type != 'Lon_Lat_Controller':
            logging.warning('mismatch between record controller Lon_Lat_Controller and configed {}'
                      .format(conf_controller_type))
            return False
    elif controller_type.debug.simple_mpc_debug:
        if conf_controller_type != 'Mpc_Controller':
            logging.warning('mismatch between record controller Mpc_Controller and configed {}'
                      .format(conf_controller_type))
            return False
    else:
        logging.warning('no controller type found in records')
        return False
    return True


def extract_data_at_multi_channels(msgs, driving_mode, gear_position):
    """Extract control/chassis/ data array and filter the control data with selected chassis features"""
    chassis_msgs = collect_message_by_topic(msgs, record_utils.CHASSIS_CHANNEL)
    control_msgs = collect_message_by_topic(msgs, record_utils.CONTROL_CHANNEL)
    localization_msgs = collect_message_by_topic(
        msgs, record_utils.LOCALIZATION_CHANNEL)
    chassis_mtx = np.array([extract_chassis_data_from_msg(msg)
                            for msg in chassis_msgs])
    control_mtx = np.array([extract_control_data_from_msg(msg)
                            for msg in control_msgs])
    localization_mtx = np.array([extract_localization_data_from_msg(msg)
                                 for msg in localization_msgs])
    logging.info('The original msgs size are: chassis {}, control {}, and localization: {}'
              .format(chassis_mtx.shape[0], control_mtx.shape[0], localization_mtx.shape[0]))
    if (chassis_mtx.shape[0] == 0 or control_mtx.shape[0] == 0 or localization_mtx.shape[0] == 0):
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
    control_idx_by_localization = np.in1d(['%.3f' % x for x in
                                           control_mtx[:, FEATURE_IDX['localization_timestamp_sec']]],
                                          ['%.3f' % x for x in
                                           localization_mtx[:, POSE_IDX['timestamp_sec']]])
    control_mtx_rtn = control_mtx[control_idx_by_chassis &
                                  control_idx_by_localization, :]
    # Third, delete the control data with inverted-sequence chassis and localization sequence_num
    # (in very rare cases, the sequence number in control record is like ... 100, 102, 101, 103 ...)
    inv_seq_chassis = (
        np.diff(control_mtx_rtn[:, FEATURE_IDX['chassis_sequence_num']]) < 0)
    inv_seq_localization = (np.diff(control_mtx_rtn[:, FEATURE_IDX['localization_timestamp_sec']])
                            < 0.0)
    control_idx_inv_seq = np.where(np.insert(inv_seq_chassis, 0, 0) |
                                   np.insert(inv_seq_localization, 0, 0))[0]
    control_mtx_rtn = np.delete(control_mtx_rtn, control_idx_inv_seq, axis=0)
    # Fourth, filter the chassis and localization data with filtered control data
    chassis_idx_rtn = []
    localization_idx_rtn = []
    chassis_idx = 0
    localization_idx = 0
    for control_idx in range(control_mtx_rtn.shape[0]):
        while (control_mtx_rtn[control_idx, FEATURE_IDX['chassis_sequence_num']] !=
               chassis_mtx_filtered[chassis_idx, MODE_IDX['sequence_num']]):
            chassis_idx += 1
        while (['%.3f' % control_mtx_rtn[control_idx, FEATURE_IDX['localization_timestamp_sec']]] !=
               ['%.3f' % localization_mtx[localization_idx, POSE_IDX['timestamp_sec']]]):
            localization_idx += 1
        chassis_idx_rtn.append(chassis_idx)
        localization_idx_rtn.append(localization_idx)
    chassis_mtx_rtn = np.take(chassis_mtx_filtered, chassis_idx_rtn, axis=0)
    localization_mtx_rtn = np.take(
        localization_mtx, localization_idx_rtn, axis=0)
    logging.info('The filtered msgs size are: chassis {}, control {}, and localization: {}'
              .format(chassis_mtx_rtn.shape[0], control_mtx_rtn.shape[0],
                      localization_mtx_rtn.shape[0]))
    # Finally, rebuild the grading mtx with the control data combined with
    # chassis and localizaiton data
    if (control_mtx_rtn.shape[0] > 0):
        # First, merge the chassis data into control data matrix
        if (chassis_mtx_rtn.shape[1] > MODE_IDX['brake_chassis']):
            grading_mtx = np.hstack((control_mtx_rtn,
                                     chassis_mtx_rtn[:, [MODE_IDX['throttle_chassis'],
                                                         MODE_IDX['brake_chassis']]]))
        else:
            grading_mtx = np.hstack((control_mtx_rtn,
                                     np.zeros((control_mtx_rtn.shape[0], 2))))
        # Second, merge the localization data into control data matrix
        pose_heading_num = np.diff(
            localization_mtx_rtn[:, POSE_IDX['pose_position_y']])
        pose_heading_den = np.diff(
            localization_mtx_rtn[:, POSE_IDX['pose_position_x']])
        sigular_idx = (pose_heading_den < MIN_EPSILON)
        pose_heading_den[sigular_idx] = 1.0
        pose_heading_offset = (np.arctan(np.divide(pose_heading_num, pose_heading_den))
                               - localization_mtx_rtn[range(localization_mtx_rtn.shape[0] - 1),
                                                      POSE_IDX['pose_heading']])
        if np.sum(np.invert(sigular_idx)) > 0:
            pose_heading_offset[sigular_idx] = np.median(
                pose_heading_offset[np.invert(sigular_idx)])
        else:
            pose_heading_offset[sigular_idx] = 0.0
        grading_mtx = np.column_stack(
            (grading_mtx, np.append(pose_heading_offset, [0.0], axis=0)))
    else:
        grading_mtx = control_mtx_rtn
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
        '/apollo/modules/data/fuel/fueling/profiling/conf/control_profiling_conf.pb.txt'
    control_profiling = ControlProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, control_profiling)
    return control_profiling


def extract_control_data_from_msg(msg):
    """Extract wanted fields from control message"""
    msg_proto = record_utils.message_to_proto(msg)
    control_latency = msg_proto.latency_stats
    control_header = msg_proto.header
    input_debug = msg_proto.debug.input_debug
    if get_config_control_profiling().controller_type == 'Lon_Lat_Controller':
        control_lon = msg_proto.debug.simple_lon_debug
        control_lat = msg_proto.debug.simple_lat_debug
        data_array = np.array([
            # Features: "Reference" category
            control_lon.station_reference,               # 0
            control_lon.speed_reference,                 # 1
            control_lon.acceleration_reference,          # 2
            control_lat.ref_heading,                     # 3
            control_lat.ref_heading_rate,                # 4
            control_lat.curvature,                       # 5
            # Features: "Error" category
            control_lon.path_remain,                     # 6
            control_lon.station_error,                   # 7
            control_lon.speed_error,                     # 8
            control_lat.lateral_error,                   # 9
            control_lat.lateral_error_rate,              # 10
            control_lat.heading_error,                   # 11
            control_lat.heading_error_rate,              # 12
            # Features: "Command" category
            msg_proto.throttle,                          # 13
            msg_proto.brake,                             # 14
            msg_proto.acceleration,                      # 15
            msg_proto.steering_target,                   # 16
            # Features: "State" category
            control_lon.current_station,                 # 17
            control_lon.current_speed,                   # 18
            control_lon.current_acceleration,            # 19
            control_lon.current_jerk,                    # 20
            control_lat.lateral_acceleration,            # 21
            control_lat.lateral_jerk,                    # 22
            control_lat.heading,                         # 23
            control_lat.heading_rate,                    # 24
            control_lat.heading_acceleration,            # 25
            control_lat.heading_jerk,                    # 26
            # Features: "Latency" category
            control_latency.total_time_ms,               # 27
            control_latency.total_time_exceeded,         # 28
            # Features: "Header" category
            control_header.timestamp_sec,                # 29
            control_header.sequence_num,                 # 30
            # Features: "Input Info" category
            input_debug.localization_header.timestamp_sec,  # 31
            input_debug.localization_header.sequence_num,   # 32
            input_debug.canbus_header.timestamp_sec,        # 33
            input_debug.canbus_header.sequence_num,         # 34
            input_debug.trajectory_header.timestamp_sec,    # 35
            input_debug.trajectory_header.sequence_num      # 36
        ])
    else:
        control_mpc = msg_proto.debug.simple_mpc_debug
        data_array = np.array([
            # Features: "Reference" category
            control_mpc.station_reference,               # 0
            control_mpc.speed_reference,                 # 1
            control_mpc.acceleration_reference,          # 2
            control_mpc.ref_heading,                     # 3
            control_mpc.ref_heading_rate,                # 4
            control_mpc.curvature,                       # 5
            # Features: "Error" category
            control_mpc.path_remain,                     # 6
            control_mpc.station_error,                   # 7
            control_mpc.speed_error,                     # 8
            control_mpc.lateral_error,                   # 9
            control_mpc.lateral_error_rate,              # 10
            control_mpc.heading_error,                   # 11
            control_mpc.heading_error_rate,              # 12
            # Features: "Command" category
            msg_proto.throttle,                          # 13
            msg_proto.brake,                             # 14
            msg_proto.acceleration,                      # 15
            msg_proto.steering_target,                   # 16
            # Features: "State" category
            control_mpc.station_feedback,                # 17
            control_mpc.speed_feedback,                  # 18
            control_mpc.acceleration_feedback,           # 19
            control_mpc.jerk_feedback,                   # 20
            control_mpc.lateral_acceleration,            # 21
            control_mpc.lateral_jerk,                    # 22
            control_mpc.heading,                         # 23
            control_mpc.heading_rate,                    # 24
            control_mpc.heading_acceleration,            # 25
            control_mpc.heading_jerk,                    # 26
            # Features: "Latency" category
            control_latency.total_time_ms,               # 27
            control_latency.total_time_exceeded,         # 28
            # Features: "Header" category
            control_header.timestamp_sec,                # 29
            control_header.sequence_num,                 # 30
            # Features: "Input Info" category
            input_debug.localization_header.timestamp_sec,  # 31
            input_debug.localization_header.sequence_num,   # 32
            input_debug.canbus_header.timestamp_sec,        # 33
            input_debug.canbus_header.sequence_num,         # 34
            input_debug.trajectory_header.timestamp_sec,    # 35
            input_debug.trajectory_header.sequence_num      # 36
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


def extract_localization_data_from_msg(msg):
    """Extract wanted fields from localization message"""
    msg_proto = record_utils.message_to_proto(msg)
    localization_header = msg_proto.header
    localization_pose = msg_proto.pose
    data_array = np.array([
        # Features: "Header" category
        localization_header.timestamp_sec,                   # 0
        localization_header.sequence_num,                    # 1
        # Features: "Pose" category
        localization_pose.position.x,                        # 2
        localization_pose.position.y,                        # 3
        localization_pose.heading                            # 4
    ])
    return data_array
