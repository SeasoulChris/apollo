#!/usr/bin/env python

import numpy as np

from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config as config
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.feature_extraction.feature_extraction_utils as utils

# The exact length of chasis messages in a group
SEGMENT_LEN = int(config['DELTA_T'] / config['delta_t'])


class InterPolationMessage(object):
    """Encapsulation of interpolation messages"""

    def __init__(self, chasis_msg=None, left_pose_msg=None, right_pose_msg=None):
        """Init, the messages here are message protos"""
        self.chasis_msg = chasis_msg
        self.left_pose_msg = left_pose_msg
        self.right_pose_msg = right_pose_msg
        self.interpolate_pose = None

    def add_pose(self, pose_msg):
        """Add a pose to the current object, need to check if it's left or right"""
        if not self.chasis_msg or not pose_msg:
            return
        if pose_msg.header.timestamp_sec <= self.chasis_msg.header.timestamp_sec:
            self.left_pose_msg = pose_msg
        elif not self.right_pose_msg:
            self.right_pose_msg = pose_msg

    def is_valid(self):
        """Check if the current object is valid"""
        # If any property is missing, it's invalid
        if not self.chasis_msg or not self.left_pose_msg or not self.right_pose_msg:
            return False
        # If left or right pose gap is too big, it's invalid
        if (self.chasis_msg.header.timestamp_sec - self.left_pose_msg.header.timestamp_sec
                > config['MAX_POSE_DELTA']
            or self.right_pose_msg.header.timestamp_sec - self.chasis_msg.header.timestamp_sec
                > config['MAX_POSE_DELTA']):
            return False
        # If any field in chasis message is np.nan, it's invalid
        chasis_data = utils.chassis_msg_to_data(self.chasis_msg)
        if any(np.isnan(feature) for feature in chasis_data):
            return False
        # If gear_location is not expected, it's invalid
        gear_Location_pos, gear_location_expected_val = len(chasis_data) - 1, 1 
        if int(chasis_data[gear_location_pos]) != gear_location_expected_val:
            return False
        # If speed is not expected, it's invalid
        speed_pos, speed_expected_val = 0, 0
        if chasis_data[speed_pos] < speed_expected_val:
            return False
        return True

    def do_interpolation(self):
        """Actually do the interpolation"""
        self.interpolate_pose = utils.interpolate_pose(self.left_pose_msg,
                                                       self.right_pose_msg,
                                                       self.chasis_msg.header.timestamp_sec)
        return self.interpolate_pose


class InterPolationMessageList(object):
    """Manages the list of InterPolationMessage, provides utils for the list"""

    def __init__(self):
        """Init"""
        self.interp_messages = []
        self.invalid_messages_count = 0

    def add_message(self, message):
        """Add a message to the list"""
        self.interp_messages.append(message)

    def find_invalid_points(self):
        """Find invalid points from the list"""
        # Set message invalid if its chasis time is too far away from last one
        invalid_intervals_count = 0
        for pos in range(1, len(self.interp_messages)):
            if (not self.interp_messages[pos].chasis_msg
                    or not self.interp_messages[pos - 1].chasis_msg):
                continue
            if (self.interp_messages[pos].chasis_msg.header.timestamp_sec
                    - self.interp_messages[pos - 1].chasis_msg.header.timestamp_sec
                    > config['CHASIS_DELTA_T']):
                invalid_intervals_count += 1
                self.interp_messages[pos].chasis_msg = None
        logging.info(F'{invalid_intervals_count}/{len(self.interp_messages)} chasis have gaps')

        invalid_messages = [i for i, x in enumerate(self.interp_messages) if not x.is_valid()]
        self.invalid_messages_count = len(invalid_messages)
        logging.info(F'{self.invalid_messages_count}/{len(self.interp_messages)} invalid messages')

        return invalid_messages

    def generate_valid_groups(self, invalid_points):
        """Generate valid groups potentially splitted by the invalid points"""
        last_pos, valid_groups = 0, []
        for pos in invalid_points:
            if last_pos < pos:
                self._add_valid_groups(valid_groups, self.interp_messages[last_pos: pos])
            last_pos = pos + 1
        if last_pos < len(self.interp_messages):
            self._add_valid_groups(valid_groups, self.interp_messages[last_pos:])
        return valid_groups

    def is_valid(self):
        """Check if the list itself is valid"""
        if not self.interp_messages:
            return False
        return (float(self.invalid_messages_count) / len(self.interp_messages)
                < config['CHASIS_DELTA_TOLERANCE_RATE'])

    def compensate_chasis(self):
        """Compensate poses where chasis messages are too close without poses in between"""
        if not config['COMPENSATE_CHASIS']:
            return
        for pos, message in enumerate(self.interp_messages):
            if message.left_pose_msg and not message.right_pose_msg:
                base_time = message.left_pose_msg.header.timestamp_sec
                message.right_pose_msg = self._find_next_right_pose(base_time, pos + 1)

    def _find_next_right_pose(self, base_time, start_pos):
        """Find next right pose message that is closest to given base_time"""
        pos, messages_len = start_pos, len(self.interp_messages)
        while pos < messages_len:
            cur_msg = self.interp_messages[pos]
            if cur_msg.left_pose_msg and cur_msg.left_pose_msg.header.timestamp_sec > base_time:
                return self.interp_messages[pos].left_pose_msg
            elif cur_msg.right_pose_msg and cur_msg.right_pose_msg.header.timestamp_sec > base_time:
                return self.interp_messages[pos].right_pose_msg
            pos += 1
        return None

    def _add_valid_groups(self, valid_groups, group):
        """Check the group and split/add it into groups if it's valid"""
        if len(group) < SEGMENT_LEN:
            return
        sub_groups = [group[idx: idx + SEGMENT_LEN]
                      for idx in range(0, len(group), SEGMENT_LEN - config['SEGMENT_OVERLAP'])]
        for sub_group in sub_groups:
            if len(sub_group) == SEGMENT_LEN:
                valid_groups.append(sub_group)
