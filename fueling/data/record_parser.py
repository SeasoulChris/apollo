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

if sys.version_info[0] >= 3:
    from cyber_py.record_py3 import RecordReader
else:
    from cyber_py.record import RecordReader

from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.gps_pb2 import Gps
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.planning.proto.planning_pb2 import ADCTrajectory

from fueling.common.coord_utils import CoordUtils
from fueling.planning.metrics.latency import LatencyMetrics
from fueling.planning.stability.speed_jerk_stability import SpeedJerkStability
from modules.data.fuel.fueling.data.proto.record_meta_pb2 import RecordMeta
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

# Configs
POS_SAMPLE_MIN_DURATION_SEC = 2
POS_SAMPLE_MIN_DISTANCE_METER = 3


def pose_distance_m(pos0, pos1):
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

        record = parser.record
        # If we have the driving_path but no map info, try guessing it.
        if record.stat.driving_path and not record.hmi_status.current_map:
            guessed_map = record_utils.guess_map_name_from_driving_path(record.stat.driving_path)
            if guessed_map:
                record.hmi_status.current_map = guessed_map
        # planning metrics
        record.stat.planning_stat.latency.max = parser._planning_latency_analyzer.get_max()
        record.stat.planning_stat.latency.min = parser._planning_latency_analyzer.get_min()
        record.stat.planning_stat.latency.avg = parser._planning_latency_analyzer.get_avg()
        for bucket, cnt in parser._planning_latency_analyzer.get_hist().items():
            record.stat.planning_stat.latency.latency_hist[bucket] = cnt

        for speed, jerk_cnt in parser._lon_stability_analyzer.get_speed_jerk_cnt().items():
            speed_jerk = record.stat.planning_stat.stability.speed_jerk.add()
            speed_jerk.speed = speed
            for jerk, cnt in jerk_cnt.items():
                speed_jerk.jerk_cnt.add(jerk=jerk, cnt=cnt)

        for speed, jerk_cnt in parser._lat_stability_analyzer.get_speed_jerk_cnt().items():
            speed_jerk = record.stat.planning_stat.stability.lat_speed_jerk.add()
            speed_jerk.speed = speed
            for jerk, cnt in jerk_cnt.items():
                speed_jerk.jerk_cnt.add(jerk=jerk, cnt=cnt)

        return record

    def __init__(self, record_file):
        """Init input reader and output record."""
        self.record = RecordMeta(path=record_file, dir=os.path.dirname(record_file))

        self._reader = RecordReader(record_file)
        # State during processing messages.
        self._current_driving_mode = None
        self._get_pose_from_gps = False
        self._last_position = None
        # To sample driving path.
        self._last_position_sampled = None
        self._last_position_sampled_time = None
        # Planning stat
        self._planning_latency_analyzer = LatencyMetrics()
        self._lon_stability_analyzer = SpeedJerkStability(is_lateral=False)
        self._lat_stability_analyzer = SpeedJerkStability(is_lateral=True)

    def ParseMeta(self):
        """
        Parse meta info which doesn't need to scan the record.
        Currently we parse the record ID, header and channel list here.
        """
        self.record.header.ParseFromString(self._reader.get_headerstring())
        for chan in self._reader.get_channellist():
            self.record.channels[chan] = self._reader.get_messagenumber(chan)
        if len(self.record.channels) == 0:
            logging.error('No message found in record')
            return False
        if (self.record.channels.get(record_utils.GNSS_ODOMETRY_CHANNEL) and
                not self.record.channels.get(record_utils.LOCALIZATION_CHANNEL)):
            logging.info('Get pose from GPS as the localization channel is missing.')
            self._get_pose_from_gps = True
        return True

    def ParseMessages(self):
        """Process all messages."""
        PROCESSORS = {
            record_utils.CHASSIS_CHANNEL: self.ProcessChassis,
            record_utils.DRIVE_EVENT_CHANNEL: self.ProcessDriveEvent,
            record_utils.GNSS_ODOMETRY_CHANNEL: self.ProcessGnssOdometry,
            record_utils.HMI_STATUS_CHANNEL: self.ProcessHMIStatus,
            record_utils.LOCALIZATION_CHANNEL: self.ProcessLocalization,
            record_utils.PLANNING_CHANNEL: self.ProcessPlanning,
        }
        for channel, msg, _type, _ in self._reader.read_messages():
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
            logging.info('Disengagement found at {}'.format(timestamp))
            disengagement = self.record.disengagements.add(time=timestamp)
            pos = self._last_position
            if pos is not None:
                try:
                    lat, lon = CoordUtils.utm_to_latlon(pos.x, pos.y)
                    disengagement.location.lat = lat
                    disengagement.location.lon = lon
                except Exception as e:
                    logging.error('Failed to parse pose to lat-lon: {}'.format(e))
        # Update DrivingMode.
        self._current_driving_mode = chassis.driving_mode

    def ProcessPlanning(self, msg):
        planning = ADCTrajectory()
        planning.ParseFromString(msg)
        self._planning_latency_analyzer.process(planning)

    def _process_position(self, time_sec, position):
        # Stat mileages.
        if self._last_position is not None:
            driving_mode = 'UNKNOWN'
            if self._current_driving_mode:
                driving_mode = Chassis.DrivingMode.Name(self._current_driving_mode)
            meters = pose_distance_m(self._last_position, position)
            if driving_mode in self.record.stat.mileages:
                self.record.stat.mileages[driving_mode] += meters
            else:
                self.record.stat.mileages[driving_mode] = meters

        # Sample driving path.
        if (self._last_position_sampled is None or
                (time_sec - self._last_position_sampled_time > POS_SAMPLE_MIN_DURATION_SEC and
                 pose_distance_m(self._last_position_sampled, position) > POS_SAMPLE_MIN_DISTANCE_METER)):
            try:
                lat, lon = CoordUtils.utm_to_latlon(position.x, position.y)
                self.record.stat.driving_path.add(lat=lat, lon=lon)
                self._last_position_sampled = position
                self._last_position_sampled_time = time_sec
            except Exception as e:
                logging.error('Failed to parse pose to lat-lon: {}'.format(e))
        # Update position.
        self._last_position = position

    def ProcessLocalization(self, msg):
        """Process Localization, stat mileages and save driving path."""
        localization = LocalizationEstimate()
        localization.ParseFromString(msg)
        # skip message from sim control that channel localization/pose contains error
        if localization.header.module_name == "SimControl" or localization.pose.position.z == 0:
            return
        self._process_position(localization.header.timestamp_sec, localization.pose.position)
        self._lon_stability_analyzer.add(localization)
        self._lat_stability_analyzer.add(localization)

    def ProcessGnssOdometry(self, msg):
        """Process GPS, stat mileages and save driving path."""
        if self._get_pose_from_gps:
            gps = Gps()
            gps.ParseFromString(msg)
            # localization initialization feature
            if gps.localization.position.z == 0:
                return
            self._process_position(gps.header.timestamp_sec, gps.localization.position)


if __name__ == '__main__':
    if len(sys.argv) > 0:
        print(RecordParser.Parse(sys.argv[-1]))
