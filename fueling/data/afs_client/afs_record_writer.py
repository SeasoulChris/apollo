#!/usr/bin/env python
"""Write record files"""

import os

from cyber.python.cyber_py3.record import RecordWriter

import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.data.afs_client.config as afs_config
import fueling.data.afs_client.conversions as conversions


class AfsRecordWriter(object):
    """Convert afs(cyber) message and write into record files with apollo format."""

    def __init__(self, target_file):
        """Init."""
        converted_file = target_file.replace(afs_config.TARGET_PATH, afs_config.CONVERT_PATH, 1)
        file_utils.makedirs(os.path.dirname(target_file))
        file_utils.makedirs(os.path.dirname(converted_file))
        self.channels = {}
        self.converted_channels = {}
        self.writer = RecordWriter(0, 0)
        self.converted_message_writer = RecordWriter(0, 0)
        self.writer.open(target_file)
        self.converted_message_writer.open(converted_file)
        conversions.register_conversions()
        logging.info(F'converted message file: {converted_file}')

    def close(self):
        """Close writers"""
        # TODO(longtao): generate real descriptor
        DESC = 'descriptor'
        for topic in self.channels:
            self.writer.write_channel(topic, self.channels[topic], DESC)
        for topic in self.converted_channels:
            self.converted_message_writer.write_channel(topic, self.converted_channels[topic], DESC)
        self.writer.close()
        self.converted_message_writer.close()

    def write_message(self, cyber_message):
        """Write message into record file"""
        if cyber_message.topic not in self.channels:
            self.channels[cyber_message.topic] = cyber_message.data_type
        self.writer.write_message(
            cyber_message.topic, cyber_message.message, cyber_message.timestamp)
        apollo_params = conversions.ConversionsCenter.convert(cyber_message)
        if not apollo_params:
            return
        apollo_topic, apollo_datatype, apollo_message = apollo_params
        if apollo_topic not in self.converted_channels:
            self.converted_channels[apollo_topic] = apollo_datatype
        self.converted_message_writer.write_message(
            apollo_topic, apollo_message.SerializeToString(), cyber_message.timestamp)
