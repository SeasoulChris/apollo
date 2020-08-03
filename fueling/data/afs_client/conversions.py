#!/usr/bin/env python
"""Manages the chain of converting methods that convert cybertron messages to apollo messages"""

from fueling.common.record.kinglong.cybertron.python.convert import transfer_localization_estimate
from fueling.common.record.kinglong.cybertron.python.convert import transfer_perception_obstacles
import fueling.common.logging as logging
import fueling.common.record.kinglong.proto.modules.localization_pose_pb2 \
    as cyber_localization_pose_pb2
import fueling.common.record.kinglong.proto.modules.perception_obstacle_pb2 \
    as cyber_perception_obstacle_pb2


class ConversionsCenter(object):
    """Conversions organizor"""
    # The static list of rules that will apply one by one
    conversions = {}

    @staticmethod
    def register(conversion):
        """Add the conversion into center"""
        ConversionsCenter.conversions[conversion.topic] = conversion

    @staticmethod
    def convert(message):
        """Convert the message by using conversion corresponding to the topic"""
        if message.topic not in ConversionsCenter.conversions:
            return None
        return ConversionsCenter.conversions[message.topic].convert(message)


class ConversionBase(object):
    """Base class for conversions"""

    def __init__(self, cyber_topic):
        """Constructor"""
        self.topic = cyber_topic
        ConversionsCenter.register(self)

    def convert(self, message):
        """Apply the conversion"""
        raise Exception('{}::convert() not implemented for base class'.format(self.topic))


class LocalizationPoseConversion(ConversionBase):
    """Convert Localization messages"""

    def __init__(self):
        """Constructor"""
        ConversionBase.__init__(self, '/localization/100hz/localization_pose')

    def convert(self, message):
        """Convert"""
        apollo_topic = '/apollo/localization/pose'
        apollo_datatype = 'apollo.localization.LocalizationEstimate'
        cyber_loc = cyber_localization_pose_pb2.LocalizationEstimate()
        cyber_loc.ParseFromString(message.message)
        apollo_loc = transfer_localization_estimate(cyber_loc)
        logging.log_every_n(
            logging.INFO, F'transferred localization: {apollo_loc.pose.position.x}', 1000)
        return (apollo_topic, apollo_datatype, apollo_loc)


class PerceptionObstaclesConversion(ConversionBase):
    """Convert PerceptionObstacle messages"""

    def __init__(self):
        """Constructor"""
        ConversionBase.__init__(self, '/perception/obstacles')

    def convert(self, message):
        """Convert"""
        apollo_topic = '/apollo/perception/obstacles'
        apollo_datatype = 'apollo.perception.PerceptionObstacles'
        cyber_obs = cyber_perception_obstacle_pb2.PerceptionObstacles()
        cyber_obs.ParseFromString(message.message)
        apollo_obs = transfer_perception_obstacles(cyber_obs.perception_obstacle)
        logging.log_every_n(logging.INFO, 'transfered perception obstacle', 1000)
        apollo_obs.header.timestamp_sec = cyber_obs.header.timestamp_sec
        return (apollo_topic, apollo_datatype, apollo_obs)


def register_conversions():
    """Register all conversions"""
    if not ConversionsCenter.conversions:
        logging.info('registering conversions')
        LocalizationPoseConversion()
        PerceptionObstaclesConversion()
