#!/usr/bin/env python

""" Vehicle dynamics feature extraction related utils. """

import os

import numpy as np

from fueling.profiling.conf.control_channel_conf import DYNAMICS_FEATURE_IDX, DYNAMICS_MODE_IDX
from modules.data.fuel.fueling.profiling.proto.control_profiling_pb2 import ControlProfiling
import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


# Message number in each segment
MSG_PER_SEGMENT = 30000
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
    conf_vehicle_type = get_profiling_config().vehicle_type
    conf_controller_type = get_profiling_config().controller_type
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


def extract_data_two_channels(msgs, driving_mode, gear_position):
    """Extract control/chassis data array and filter the control data with selected chassis features"""
    chassis_msgs = collect_message_by_topic(msgs, record_utils.CHASSIS_CHANNEL)
    control_msgs = collect_message_by_topic(msgs, record_utils.CONTROL_CHANNEL)

    chassis_mtx = np.array([extract_chassis_data_from_msg(msg)
                            for msg in chassis_msgs])
    control_mtx = np.array([extract_control_data_from_msg(msg)
                            for msg in control_msgs])

    logging.info('The original msgs size are: chassis {}, control {}'
                 .format(chassis_mtx.shape[0], control_mtx.shape[0]))
    if (chassis_mtx.shape[0] == 0 or control_mtx.shape[0] == 0):
        return np.take(control_mtx, [], axis=0)
    # First, filter the chassis data with desired driving modes and gear locations
    driving_condition = (
        chassis_mtx[:, DYNAMICS_MODE_IDX['driving_mode']] == driving_mode)
    gear_condition = (
        chassis_mtx[:, DYNAMICS_MODE_IDX['gear_location']] == gear_position[0])
    for gear_idx in range(1, len(gear_position)):
        gear_condition |= (
            chassis_mtx[:, DYNAMICS_MODE_IDX['gear_location']] == gear_position[gear_idx])
    chassis_idx_filtered = np.where(driving_condition & gear_condition)[0]
    chassis_mtx_filtered = np.take(chassis_mtx, chassis_idx_filtered, axis=0)
    # Second, filter the control data with existing chassis and localization sequence_num
    control_idx_by_chassis = np.in1d(control_mtx[:, DYNAMICS_FEATURE_IDX['chassis_sequence_num']],
                                     chassis_mtx_filtered[:, DYNAMICS_MODE_IDX['sequence_num']])
    control_mtx_rtn = control_mtx[control_idx_by_chassis, :]
    # Third, swap the control data with inverted-sequence chassis sequence_num
    # (in very rare cases, the chassis sequence number in record is like ... 100, 102, 101, 103 ...)
    for inv in np.where(
            np.diff(control_mtx_rtn[:, DYNAMICS_FEATURE_IDX['chassis_sequence_num']]) < 0):
        control_mtx_rtn[[inv, inv + 1], :] = control_mtx_rtn[[inv + 1, inv], :]
    for inv in np.where(np.diff(chassis_mtx_filtered[:, DYNAMICS_MODE_IDX['sequence_num']]) < 0):
        chassis_mtx_filtered[[inv, inv + 1], :] = chassis_mtx_filtered[[inv + 1, inv], :]
    # Fourth, filter the chassis data with filtered control data
    chassis_idx_rtn = []
    chassis_idx = 0
    for control_idx in range(control_mtx_rtn.shape[0]):
        while (control_mtx_rtn[control_idx, DYNAMICS_FEATURE_IDX['chassis_sequence_num']] !=
               chassis_mtx_filtered[chassis_idx, DYNAMICS_MODE_IDX['sequence_num']]):
            chassis_idx += 1
        chassis_idx_rtn.append(chassis_idx)
    chassis_mtx_rtn = np.take(chassis_mtx_filtered, chassis_idx_rtn, axis=0)
    logging.info('The filtered msgs size are: chassis {}, control {}'
                 .format(chassis_mtx_rtn.shape[0], control_mtx_rtn.shape[0]))
    # Finally, rebuild the grading mtx with the control data combined with chassis data
    # TODO(fengzongbao) Filter acceleration_reference by positive and negative to throttle and brake
    if (control_mtx_rtn.shape[0] > 0):
        # Merge the chassis data into control data matrix
        if (chassis_mtx_rtn.shape[1] > DYNAMICS_MODE_IDX['brake_chassis']):
            grading_mtx = np.hstack((control_mtx_rtn,
                                     chassis_mtx_rtn[:, [DYNAMICS_MODE_IDX['throttle_chassis'],
                                                         DYNAMICS_MODE_IDX['brake_chassis']]]))
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
            msg_proto.throttle,                             # 1
            msg_proto.brake,                                # 2
            msg_proto.acceleration,                         # 3
            msg_proto.steering_target,                      # 4
            # Features: "Reference" category
            control_lon.current_acceleration,               # 5
            control_lat.steering_position,                  # 6
            # Features: "Header" category
            control_header.timestamp_sec,                   # 7
            control_header.sequence_num,                    # 8
            # Features: "Input Info" category
            input_debug.canbus_header.timestamp_sec,        # 9
            input_debug.canbus_header.sequence_num,         # 10
        ])
    else:
        control_mpc = msg_proto.debug.simple_mpc_debug
        data_array = np.array([
            # Features: "Command" category
            msg_proto.throttle,                             # 1
            msg_proto.brake,                                # 2
            msg_proto.acceleration,                         # 3
            msg_proto.steering_target,                      # 4
            # Features: "State" category
            control_mpc.acceleration_feedback,              # 5
            control_mpc.steering_position,                  # 6
            # Features: "Header" category
            control_header.timestamp_sec,                   # 7
            control_header.sequence_num,                    # 8
            # Features: "Input Info" category
            input_debug.canbus_header.timestamp_sec,        # 9
            input_debug.canbus_header.sequence_num          # 10
        ])

    return data_array


def extract_chassis_data_from_msg(msg):
    """Extract wanted fields from chassis message"""
    msg_proto = record_utils.message_to_proto(msg)
    chassis_header = msg_proto.header
    if (get_profiling_config().vehicle_type.find('Mkz') or
            get_profiling_config().vehicle_type.find('Lexus')):
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
