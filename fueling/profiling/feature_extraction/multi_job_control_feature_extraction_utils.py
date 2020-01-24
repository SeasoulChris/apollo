#!/usr/bin/env python

""" Control feature extraction related utils. """

import os

import numpy as np

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, MODE_IDX, POSE_IDX
from modules.data.fuel.fueling.profiling.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.common.sanity_check as sanity_check
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils

# Message number in each segment
MSG_PER_SEGMENT = 3000
# Maximum allowed time gap betwee two messages
MAX_PHASE_DELTA = 0.01
# Minimum epsilon value used in compare with zero
MIN_EPSILON = 0.000001


CONF_FILE = 'vehicle_param.pb.txt'
CHANNELS = {record_utils.CHASSIS_CHANNEL,
            record_utils.LOCALIZATION_CHANNEL, record_utils.CONTROL_CHANNEL}


def parse_vehicle_controller(task, flags):
    """Parse controller/vehicle type by task"""
    if not os.path.exists(task):
        logging.warning(F'task directory is invalid: {task}')
        return False
    record_file = next((os.path.join(task, record_file) for record_file in os.listdir(task)
                        if (record_utils.is_record_file(record_file) or
                            record_utils.is_bag_file(record_file))), None)
    if not record_file:
        logging.warning(F'no valid record file found in task: {task}')
        return False
    # Read two topics together to avoid looping all messages in the record file twice
    logging.info(F'Parse vehicle controller in task {task}, record {record_file}')
    record_dir_splited = task.split('/')
    record_prefix = '{}/{}'.format(
        record_dir_splited[-2], record_dir_splited[-1])
    read_record_func = record_utils.read_record([record_utils.CONTROL_CHANNEL,
                                                 record_utils.HMI_STATUS_CHANNEL])
    messages = read_record_func(record_file)
    logging.info(F'{len(messages)} messages for record file {record_file}')
    vehicle_message = get_message_by_topic(
        messages, record_utils.HMI_STATUS_CHANNEL)
    if vehicle_message and hasattr(
            record_utils.message_to_proto(vehicle_message), 'current_vehicle'):
        vehicle_type = record_utils.message_to_proto(
            vehicle_message).current_vehicle
    elif flags['ctl_metrics_simulation_only_test']:
        vehicle_type = flags['ctl_metrics_simulation_vehicle']
    else:
        logging.info(F'no vehicle messages found in task {task} record {record_file}; '
                     F'use "Arbitrary" as the current vehicle type')
        vehicle_type = "Arbitrary"
    # Compare the vehicle_type from HMI_status channel and the vehicle_type from the input file system.
    vehicle_type_parsed_from_dir = multi_vehicle_utils.get_vehicle_by_task(task)
    if vehicle_type == "Arbitrary":
        vehicle_type = vehicle_type_parsed_from_dir
    else:
        if vehicle_type.lower() != vehicle_type_parsed_from_dir.lower():
            logging.info(F'vehicle type parsed from hmi_status {vehicle_type} '
                         F'does not match input data path {vehicle_type_parsed_from_dir}, '
                         F'we will process by vehicle type from hmi_status')
    control_message = get_message_by_topic(
        messages, record_utils.CONTROL_CHANNEL)
    # Check control important field
    if control_message and hasattr(record_utils.message_to_proto(control_message), 'debug'):
        if (hasattr(record_utils.message_to_proto(control_message).debug, 'simple_lon_debug') and
                hasattr(record_utils.message_to_proto(control_message).debug, 'simple_lat_debug')):
            controller_type = "Lon_Lat_Controller"
        elif hasattr(record_utils.message_to_proto(control_message).debug, 'simple_mpc_debug'):
            controller_type = "MPC_Controller"
        else:
            logging.info(F'no known controller type found in task {task} record {record_file}; '
                         F'use "Arbitrary" as the current controller type')
            controller_type = "Arbitrary"
    else:
        logging.warning(F'no control messages found in task {task} record {record_file}; '
                        F'stop control profiling procedure')
        return False
    vehicle_controller_parsed_dir = '{}/{}/{}'.format(vehicle_type, controller_type, record_prefix)
    return (vehicle_controller_parsed_dir, task)


def extract_data_at_multi_channels(msgs, flags, driving_mode, gear_position):
    """Extract control/chassis/ data array and filter the control data with selected chassis features"""
    chassis_msgs = collect_message_by_topic(msgs, record_utils.CHASSIS_CHANNEL)
    control_msgs = collect_message_by_topic(msgs, record_utils.CONTROL_CHANNEL)
    localization_msgs = collect_message_by_topic(
        msgs, record_utils.LOCALIZATION_CHANNEL)
    chassis_mtx = np.array([data for data in [extract_chassis_data_from_msg(msg)
                                              for msg in chassis_msgs] if data is not None])
    control_mtx = np.array([data for data in [extract_control_data_from_msg(msg)
                                              for msg in control_msgs] if data is not None])
    localization_mtx = np.array([data for data in [extract_localization_data_from_msg(msg)
                                                   for msg in localization_msgs] if data is not None])
    logging.info(F'The original msgs size are: chassis {chassis_mtx.shape[0]}, '
                 F'control {control_mtx.shape[0]}, and localization {localization_mtx.shape[0]}')
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
    control_idx_by_chassis = np.in1d(['%.3f' % x for x in
                                      control_mtx[:, FEATURE_IDX['chassis_timestamp_sec']]],
                                     ['%.3f' % x for x in
                                      chassis_mtx_filtered[:, MODE_IDX['timestamp_sec']]])
    control_idx_by_localization = np.in1d(['%.3f' % x for x in
                                           control_mtx[:, FEATURE_IDX['localization_timestamp_sec']]],
                                          ['%.3f' % x for x in
                                           localization_mtx[:, POSE_IDX['timestamp_sec']]])
    control_mtx_rtn = control_mtx[control_idx_by_chassis &
                                  control_idx_by_localization, :]
    # Third, filter the control data with specific status flags
    if flags['ctl_metrics_filter_by_MRAC']:
        MRAC_condition = (control_mtx_rtn[:, FEATURE_IDX['steer_mrac_enable_status']] != 1)
        control_mtx_rtn = np.delete(control_mtx_rtn, np.where(MRAC_condition)[0], axis=0)
        logging.info(F'Control channel messages with MRAC off are filtered out')
    # Fourth, delete the control data with inverted-sequence chassis and localization sequence_num
    # (in very rare cases, the sequence number in control record is like ... 100, 102, 101, 103 ...)
    inv_seq_chassis = (np.diff(control_mtx_rtn[:, FEATURE_IDX['chassis_timestamp_sec']]) < 0.0)
    inv_seq_localization = (np.diff(control_mtx_rtn[:, FEATURE_IDX['localization_timestamp_sec']])
                            < 0.0)
    for inv in np.where(inv_seq_chassis | inv_seq_localization):
        control_mtx_rtn = np.delete(control_mtx_rtn, [inv, inv + 1], axis=0)
    # Fourth, filter the chassis and localization data with filtered control data
    chassis_idx_rtn = []
    localization_idx_rtn = []
    chassis_idx = 0
    localization_idx = 0
    for control_idx in range(control_mtx_rtn.shape[0]):
        chassis_timestamp = round(control_mtx_rtn[control_idx,
                                                  FEATURE_IDX['chassis_timestamp_sec']], 3)
        localization_timestamp = round(
            control_mtx_rtn[control_idx, FEATURE_IDX['localization_timestamp_sec']], 3)
        while (round(chassis_mtx_filtered[chassis_idx, MODE_IDX['timestamp_sec']], 3) !=
               chassis_timestamp):
            chassis_idx += 1
        chassis_idx_rtn.append(chassis_idx)
        while (round(localization_mtx[localization_idx, POSE_IDX['timestamp_sec']], 3) !=
               localization_timestamp):
            localization_idx += 1
        localization_idx_rtn.append(localization_idx)
    chassis_mtx_rtn = np.take(chassis_mtx_filtered, chassis_idx_rtn, axis=0)
    localization_mtx_rtn = np.take(
        localization_mtx, localization_idx_rtn, axis=0)
    logging.info(F'The filtered msgs size are: chassis {chassis_mtx_rtn.shape[0]}, '
                 F'control {control_mtx_rtn.shape[0]}, and '
                 F'localization {localization_mtx_rtn.shape[0]}')
    # Finally, rebuild the grading mtx with the control data combined with
    # chassis and localizaiton data
    if (control_mtx_rtn.shape[0] > 0):
        # First, merge the planning-related data into control data matrix
        replan_flag = np.multiply(np.diff(
            control_mtx_rtn[:, FEATURE_IDX['replan_sequence_num']]) > 0, 1)
        grading_mtx = np.column_stack(
            (control_mtx_rtn, np.append(replan_flag, [0], axis=0)))
        # Second, merge the chassis data into control data matrix
        grading_mtx = np.hstack((grading_mtx,
                                 chassis_mtx_rtn[:, [MODE_IDX['throttle_chassis'],
                                                     MODE_IDX['brake_chassis'],
                                                     MODE_IDX['steering_chassis']]]))
        # Third, merge the localization data into control data matrix
        grading_mtx = np.hstack((grading_mtx,
                                 localization_mtx_rtn[:, [POSE_IDX['pose_position_x'],
                                                          POSE_IDX['pose_position_y'],
                                                          POSE_IDX['pose_heading']]]))
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
    control_latency = getattr(msg_proto, 'latency_stats', float('NaN'))
    control_header = getattr(msg_proto, 'header', float('NaN'))
    input_debug = getattr(getattr(msg_proto, 'debug', float('NaN')),
                          'input_debug', float('NaN'))
    if (hasattr(getattr(msg_proto, 'debug'), 'simple_lon_debug') and
            hasattr(getattr(msg_proto, 'debug'), 'simple_lat_debug')):
        control_lon = msg_proto.debug.simple_lon_debug
        control_lat = msg_proto.debug.simple_lat_debug
        data_array = np.array([
            # Features: "Reference" category
            getattr(control_lon, 'station_reference', float('NaN')),               # 0
            getattr(control_lon, 'speed_reference', float('NaN')),                 # 1
            getattr(control_lon, 'acceleration_reference', float('NaN')),          # 2
            getattr(control_lat, 'ref_heading', float('NaN')),                     # 3
            getattr(control_lat, 'ref_heading_rate', float('NaN')),                # 4
            getattr(control_lat, 'curvature', float('NaN')),                       # 5
            # Features: "Error" category
            getattr(control_lon, 'path_remain', float('NaN')),                     # 6
            getattr(control_lon, 'station_error', float('NaN')),                   # 7
            getattr(control_lon, 'speed_error', float('NaN')),                     # 8
            getattr(control_lat, 'lateral_error', float('NaN')),                   # 9
            getattr(control_lat, 'lateral_error_rate', float('NaN')),              # 10
            getattr(control_lat, 'heading_error', float('NaN')),                   # 11
            getattr(control_lat, 'heading_error_rate', float('NaN')),              # 12
            # Features: "Command" category
            getattr(msg_proto, 'throttle', float('NaN')),                          # 13
            getattr(msg_proto, 'brake', float('NaN')),                             # 14
            getattr(msg_proto, 'acceleration', float('NaN')),                      # 15
            getattr(msg_proto, 'steering_target', float('NaN')),                   # 16
            # Features: "State" category
            getattr(control_lon, 'current_station', float('NaN')),                 # 17
            getattr(control_lon, 'current_speed', float('NaN')),                   # 18
            getattr(control_lon, 'current_acceleration', float('NaN')),            # 19
            getattr(control_lon, 'current_jerk', float('NaN')),                    # 20
            getattr(control_lat, 'lateral_acceleration', float('NaN')),            # 21
            getattr(control_lat, 'lateral_jerk', float('NaN')),                    # 22
            getattr(control_lat, 'heading', float('NaN')),                         # 23
            getattr(control_lat, 'heading_rate', float('NaN')),                    # 24
            getattr(control_lat, 'heading_acceleration', float('NaN')),            # 25
            getattr(control_lat, 'heading_jerk', float('NaN')),                    # 26
            # Features: "Pose" category
            getattr(getattr(getattr(control_lon, 'current_reference_point',
                                    float('NaN')), 'path_point', float('NaN')),
                    'x', float('NaN')),                                            # 27
            getattr(getattr(getattr(control_lon, 'current_reference_point',
                                    float('NaN')), 'path_point', float('NaN')),
                    'y', float('NaN')),                                            # 28
            # Features: "Status" category
            getattr(control_lat, 'steer_mrac_enable_status', float('NaN')),        # 29
            # Features: "Latency" category
            getattr(control_latency, 'total_time_ms', float('NaN')),               # 30
            getattr(control_latency, 'total_time_exceeded', float('NaN')),         # 31
            # Features: "Header" category
            getattr(control_header, 'timestamp_sec', float('NaN')),                # 32
            getattr(control_header, 'sequence_num', float('NaN')),                 # 33
            # Features: "Input Info" category
            getattr(getattr(input_debug, 'localization_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 34
            getattr(getattr(input_debug, 'localization_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 35
            getattr(getattr(input_debug, 'canbus_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 36
            getattr(getattr(input_debug, 'canbus_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 37
            getattr(getattr(input_debug, 'trajectory_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 38
            getattr(getattr(input_debug, 'trajectory_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 39
            getattr(getattr(input_debug, 'latest_replan_trajectory_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 40
            getattr(getattr(input_debug, 'latest_replan_trajectory_header', float('NaN')),
                    'sequence_num', float('NaN'))                                  # 41
        ])
    elif hasattr(getattr(msg_proto, 'debug'), 'simple_mpc_debug'):
        control_mpc = msg_proto.debug.simple_mpc_debug
        data_array = np.array([
            # Features: "Reference" category
            getattr(control_mpc, 'station_reference', float('NaN')),               # 0
            getattr(control_mpc, 'speed_reference', float('NaN')),                 # 1
            getattr(control_mpc, 'acceleration_reference', float('NaN')),          # 2
            getattr(control_mpc, 'ref_heading', float('NaN')),                     # 3
            getattr(control_mpc, 'ref_heading_rate', float('NaN')),                # 4
            getattr(control_mpc, 'curvature', float('NaN')),                       # 5
            # Features: "Error" category
            getattr(control_mpc, 'path_remain', float('NaN')),                     # 6
            getattr(control_mpc, 'station_error', float('NaN')),                   # 7
            getattr(control_mpc, 'speed_error', float('NaN')),                     # 8
            getattr(control_mpc, 'lateral_error', float('NaN')),                   # 9
            getattr(control_mpc, 'lateral_error_rate', float('NaN')),              # 10
            getattr(control_mpc, 'heading_error', float('NaN')),                   # 11
            getattr(control_mpc, 'heading_error_rate', float('NaN')),              # 12
            # Features: "Command" category
            getattr(msg_proto, 'throttle', float('NaN')),                          # 13
            getattr(msg_proto, 'brake', float('NaN')),                             # 14
            getattr(msg_proto, 'acceleration', float('NaN')),                      # 15
            getattr(msg_proto, 'steering_target', float('NaN')),                   # 16
            # Features: "State" category
            getattr(control_mpc, 'station_feedback', float('NaN')),                # 17
            getattr(control_mpc, 'speed_feedback', float('NaN')),                  # 18
            getattr(control_mpc, 'acceleration_feedback', float('NaN')),           # 19
            getattr(control_mpc, 'jerk_feedback', float('NaN')),                   # 20
            getattr(control_mpc, 'lateral_acceleration', float('NaN')),            # 21
            getattr(control_mpc, 'lateral_jerk', float('NaN')),                    # 22
            getattr(control_mpc, 'heading', float('NaN')),                         # 23
            getattr(control_mpc, 'heading_rate', float('NaN')),                    # 24
            getattr(control_mpc, 'heading_acceleration', float('NaN')),            # 25
            getattr(control_mpc, 'heading_jerk', float('NaN')),                    # 26
            # Features: "Pose" category
            getattr(getattr(getattr(control_mpc, 'current_reference_point',
                                    float('NaN')), 'path_point', float('NaN')),
                    'x', float('NaN')),                                            # 27
            getattr(getattr(getattr(control_mpc, 'current_reference_point',
                                    float('NaN')), 'path_point', float('NaN')),
                    'y', float('NaN')),                                            # 28
            # Features: "Status" category
            getattr(control_mpc, 'steer_mrac_enable_status', float('NaN')),        # 29
            # Features: "Latency" category
            getattr(control_latency, 'total_time_ms', float('NaN')),               # 30
            getattr(control_latency, 'total_time_exceeded', float('NaN')),         # 31
            # Features: "Header" category
            getattr(control_header, 'timestamp_sec', float('NaN')),                # 32
            getattr(control_header, 'sequence_num', float('NaN')),                 # 33
            # Features: "Input Info" category
            getattr(getattr(input_debug, 'localization_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 34
            getattr(getattr(input_debug, 'localization_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 35
            getattr(getattr(input_debug, 'canbus_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 36
            getattr(getattr(input_debug, 'canbus_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 37
            getattr(getattr(input_debug, 'trajectory_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 38
            getattr(getattr(input_debug, 'trajectory_header', float('NaN')),
                    'sequence_num', float('NaN')),                                 # 39
            getattr(getattr(input_debug, 'latest_replan_trajectory_header', float('NaN')),
                    'timestamp_sec', float('NaN')),                                # 40
            getattr(getattr(input_debug, 'latest_replan_trajectory_header', float('NaN')),
                    'sequence_num', float('NaN'))                                  # 41
        ])
    else:
        # Return None for Non-recognized Controller Type
        return None
    return data_array


def extract_chassis_data_from_msg(msg):
    """Extract wanted fields from chassis message"""
    msg_proto = record_utils.message_to_proto(msg)
    chassis_header = getattr(msg_proto, 'header', float('NaN'))
    data_array = np.array([
        # Features: "Status" category
        getattr(msg_proto, 'driving_mode', float('NaN')),                          # 0
        getattr(msg_proto, 'gear_location', float('NaN')),                         # 1
        # Features: "Header" category
        getattr(chassis_header, 'timestamp_sec', float('NaN')),                    # 2
        getattr(chassis_header, 'sequence_num', float('NaN')),                     # 3
        # Features: "Action" category
        getattr(msg_proto, 'throttle_percentage', float('NaN')),                   # 4
        getattr(msg_proto, 'brake_percentage', float('NaN')),                      # 5
        getattr(msg_proto, 'steering_percentage', float('NaN'))                    # 6
    ])
    return data_array


def extract_localization_data_from_msg(msg):
    """Extract wanted fields from localization message"""
    msg_proto = record_utils.message_to_proto(msg)
    localization_header = getattr(msg_proto, 'header', float('NaN'))
    localization_pose = getattr(msg_proto, 'pose', float('NaN'))
    data_array = np.array([
        # Features: "Header" category
        getattr(localization_header, 'timestamp_sec', float('NaN')),                   # 0
        getattr(localization_header, 'sequence_num', float('NaN')),                    # 1
        # Features: "Pose" category
        getattr(getattr(localization_pose, 'position',
                        float('NaN')), 'x', float('NaN')),                             # 2
        getattr(getattr(localization_pose, 'position',
                        float('NaN')), 'y', float('NaN')),                             # 3
        getattr(localization_pose, 'heading', float('NaN'))                            # 4
    ])
    return data_array
