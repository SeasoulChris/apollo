#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""
Parse Cyber record into apollo.data.Record.

Use as command tool: record_parser.py <record>
Use as util lib:     RecordParser.Parse(<record>)
"""

import math
import os
import sys

import colored_glog as glog
import utm

from cyber_py.record import RecordReader
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate

from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


# Configs
POS_SAMPLE_MIN_DURATION_SEC = 2
POS_SAMPLE_MIN_DISTANCE_METER = 3
UTM_ZONE_ID = 10
UTM_ZONE_LETTER = 'S'


def utm_distance_m(pos0, pos1):
    """Return distance of pos0 and pos1 in meters."""
    return math.sqrt((pos0.x - pos1.x) ** 2 + (pos0.y - pos1.y) ** 2 + (pos0.z - pos1.z) ** 2)


class RecordParser(object):
    """Wrapper of a Cyber record."""

    @staticmethod
    def Parse(record_file):
        """Simple interface to parse a cyber record."""
        parser = RecordParser(record_file)
        if not parser.ParseMeta():
            return None
        parser.ParseMessages()
        return parser.record

    def __init__(self, record_file):
        """Init input reader and output record."""
        self.record = RecordMeta(path=record_file, dir=os.path.dirname(record_file))

        self._reader = RecordReader(s3_utils.abs_path(record_file))
        # State during processing messages.
        self._current_driving_mode = None
        self._last_position = None
        # To sample driving path.
        self._last_position_sampled = None
        self._last_position_sampled_time = None

    def ParseMeta(self):
        """
        Parse meta info which doesn't need to scan the record.
        Currently we parse the record ID, header and channel list here.
        """
        self.record.header.ParseFromString(self._reader.get_headerstring())
        for chan in self._reader.get_channellist():
            self.record.channels[chan] = self._reader.get_messagenumber(chan)
        if len(self.record.channels) == 0:
            glog.error('No message found in record')
            return False
        return True

    def ParseMessages(self):
        """Process all messages."""
        PROCESSORS = {
            record_utils.CHASSIS_CHANNEL: self.ProcessChassis,
            record_utils.DRIVE_EVENT_CHANNEL: self.ProcessDriveEvent,
            record_utils.HMI_STATUS_CHANNEL: self.ProcessHMIStatus,
            record_utils.LOCALIZATION_CHANNEL: self.ProcessLocalization,
        }
        for channel, msg, _type, timestamp in self._reader.read_messages():
            processor = PROCESSORS.get(channel)
            processor(msg) if processor else None

    def ProcessHMIStatus(self, msg):
        """Save HMIStatus."""
        # Keep the first message and assume it doesn't change in one recording.
        if not self.record.HasField('hmi_status'):
            self.record.hmi_status.ParseFromString(msg)

    def ProcessDriveEvent(self, msg):
        """Save DriveEvents."""
        self.record.drive_events.add().ParseFromString(msg)

    def ProcessChassis(self, msg):
        """Process Chassis, save disengagements."""
        chassis = Chassis()
        chassis.ParseFromString(msg)
        timestamp = chassis.header.timestamp_sec
        if self._current_driving_mode == chassis.driving_mode:
            # DrivingMode doesn't change.
            return
        # Save disengagement.
        if (self._current_driving_mode == Chassis.COMPLETE_AUTO_DRIVE and
            chassis.driving_mode == Chassis.EMERGENCY_MODE):
            glog.info('Disengagement found at {}'.format(timestamp))
            disengagement = self.record.disengagements.add(time=timestamp)
            if self._last_position is not None:
                lat, lon = utm.to_latlon(self._last_position.x, self._last_position.y,
                                         UTM_ZONE_ID, UTM_ZONE_LETTER)
                disengagement.location.lat = lat
                disengagement.location.lon = lon
        # Update DrivingMode.
        self._current_driving_mode = chassis.driving_mode

    def ProcessLocalization(self, msg):
        """Process Localization, stat mileages and save driving path."""
        localization = LocalizationEstimate()
        localization.ParseFromString(msg)
        timestamp = localization.header.timestamp_sec
        cur_pos = localization.pose.position

        # Stat mileages.
        if self._last_position is not None and self._current_driving_mode is not None:
            driving_mode = Chassis.DrivingMode.Name(self._current_driving_mode)
            meters = utm_distance_m(self._last_position, cur_pos)
            if driving_mode in self.record.stat.mileages:
                self.record.stat.mileages[driving_mode] += meters
            else:
                self.record.stat.mileages[driving_mode] = meters

        # Sample driving path.
        if (self._last_position_sampled is None or
            (timestamp - self._last_position_sampled_time > POS_SAMPLE_MIN_DURATION_SEC and
             utm_distance_m(self._last_position_sampled, cur_pos) > POS_SAMPLE_MIN_DISTANCE_METER)):
            self._last_position_sampled = cur_pos
            self._last_position_sampled_time = timestamp
            lat, lon = utm.to_latlon(cur_pos.x, cur_pos.y, UTM_ZONE_ID, UTM_ZONE_LETTER)
            self.record.stat.driving_path.add(lat=lat, lon=lon)
        # Update position.
        self._last_position = cur_pos


if __name__ == '__main__':
    if len(sys.argv) > 0:
        print(RecordParser.Parse(sys.argv[-1]))
