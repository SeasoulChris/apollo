# ****************************************************************************
# Copyright 2017 The Apollo Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ****************************************************************************
# -*- coding: utf-8 -*-

import collections
import sys

from google.protobuf.descriptor_pb2 import FileDescriptorProto

import fueling.common.record.kinglong.cybertron.python.cyber_bag as cyber_bag

PyBagMessage = collections.namedtuple('PyBagMessage', 'topic message data_type timestamp')

class RecordException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

class RecordUnindexException(RecordException):
    def __init__(self):
        RecordException.__init__(self, "Unindexed record")

class RecordInvalidException(RecordException):
    def __init__(self):
        RecordException.__init__(self, "Invalid record")

class Bag:
    """
    Class for cybertron Node wrapper.
    """
    def __init__(self, name, write_mode=False, if_dump_parameter_snapshot=False):
        """
        @param self
        @param name str: node name
        """
        self.start_timestamp = None
        self.end_timestamp = None
        self.write_mode = False
        if write_mode:
            self.bag = cyber_bag.PyBag(name, True, if_dump_parameter_snapshot)
            self.write_mode = True
        else:
            # set file meta inf
            self.bag = cyber_bag.PyBag(name, False, if_dump_parameter_snapshot)
            if not self.bag.is_valid():
                raise RecordInvalidException()
            if self.bag.is_active():
                raise RecordUnindexException()
            self.start_timestamp = self.bag.get_start_time() / 1000000000.0
            self.end_timestamp = self.bag.get_end_time() / 1000000000.0
            self.topics = self.bag.get_channels()
            self.file_size = self.bag.get_file_size()
            self.messages = self.bag.get_message_count()
            self.file_name = self.bag.get_file_name()
            self.version = self.bag.get_version()
            self.compress_type = self.bag.get_compress_type()

    def read_messages(self, topics = [], start_time=0, end_time=0):
        """
        read message from bag file.
        @param self
        @param topics list: topic list
        @param start_time: 
        @param end_time:
        @return: generator of (topic, message, data_type, timestamp) namedtuples for each message in the bag file
        """
        while True:
            message = self.bag.read(topics, start_time, end_time)
            if not message.end:
                yield PyBagMessage(message.topic, message.data, message.data_type, message.timestamp)
            else:
                #print "No message more."
                break

    def write(self, topic, data, data_class, t = 0, raw = True):
        """
        create a topic reader for receive message from topic.
        @param self
        @param topic str: topic name
        @param data str or proto: message data
        @t : timestamp of the message
        @raw bool:  
        @data_class proto:
        """
        if not raw:
            datatype = data.DESCRIPTOR.full_name
            self.register_message(data.DESCRIPTOR.file)
            self.bag.write(topic, data.SerializeToString(), datatype, t)
        else:
            self.bag.write(topic, data, str(data_class), t)

    def register_message(self, file_desc):
        for dep in file_desc.dependencies:
            self.register_message(dep)
        proto = FileDescriptorProto()
        file_desc.CopyToProto(proto)
        proto.name = file_desc.name
        desc_str = proto.SerializeToString()
        self.bag.register_message(desc_str)

    def get_desc(self, name):
        return self.bag.get_desc(name)

    def set_desc(self, name, msg_type, desc):
        self.bag.set_desc(name, msg_type, desc)

    def get_snapshot(self):
        return self.bag.get_snapshot()

    def set_snapshot(self, snapshot):
        self.bag.set_snapshot(snapshot)

    def reset(self):
        """
        reset iterator of read bag message
        """
        self.bag.reset()

    def get_message_count(self, topic = None):
        """
        get message count of the bag file or topic in the bag file.
        """
        if topic is None:
            return self.bag.get_message_count()
        else:
            return self.bag.get_message_count(topic)

    def get_start_time(self, topic = None):
        """
        get start time of the bag file or topic in the bag file if specify topic name.
        @param topic
        """
        if topic is None:
            return self.bag.get_start_time()
        else:
            return self.bag.get_start_time(topic)

    def get_end_time(self, topic = None):
        """
        get end time of the bag file or topic in the bag file if specify topic name.
        @param self
        """
        if topic is None:
            return self.bag.get_end_time()
        else:
            return self.bag.get_end_time(topic)

    def get_file_name(self):
        """
        get file name.
        @param self
        """
        return self.bag.get_file_name()

    def get_file_size(self):
        """
        get file size.
        @param self
        """
        return self.bag.get_file_size()

    def get_yaml_info(self, key=None):
        s = ''

        try:
            s += 'path: %s\n' % self.file_name
            s += 'version: %s\n' % (self.version)

            start_stamp = self.start_timestamp
            end_stamp   = self.end_timestamp

            duration = end_stamp - start_stamp
            s += 'duration: %.6f\n' % duration
            s += 'start: %.6f\n' % start_stamp
            s += 'end: %.6f\n' % end_stamp
            s += 'size: %d\n' % self.file_size
            s += 'messages: %d\n' % self.messages
            #s += 'indexed: True\n'

            if self.compress_type == 0:
                s += 'compression: none\n'
            else:
                s += 'compression: %s\n' % str('bzip2')

            s += 'types:\n'
            datatypes = []
            for topic in self.topics:
                data_type = self.bag.get_channel_type(topic)
                msg_num = self.bag.get_message_count(topic)
                datatypes.append((topic, data_type, msg_num))
                s += '    - type: %s\n' % data_type

            # Show topics
            s += 'topics:\n'
            for c in datatypes:
                topic_msg_count = c[2]
                s += '    - topic: %s\n' % c[0]
                s += '      type: %s\n' % c[1]
                s += '      messages: %d\n' % topic_msg_count

            if not key:
                return s

            class DictObject(object):
                def __init__(self, d):
                    for a, b in d.items():
                        if isinstance(b, (list, tuple)):
                           setattr(self, a, [DictObject(x) if isinstance(x, dict) else x for x in b])
                        else:
                           setattr(self, a, DictObject(b) if isinstance(b, dict) else b)

            obj = DictObject(yaml.load(s))
            try:
                val = eval('obj.' + key)
            except Exception as ex:
                print('Error getting key "%s"' % key)
                return None

            def print_yaml(val, indent=0):
                indent_str = '  ' * indent

                if type(val) is list:
                    s = ''
                    for item in val:
                        s += '%s- %s\n' % (indent_str, print_yaml(item, indent + 1))
                    return s
                elif type(val) is DictObject:
                    s = ''
                    for i, (k, v) in enumerate(val.__dict__.items()):
                        if i != 0:
                            s += indent_str
                        s += '%s: %s' % (k, str(v))
                        if i < len(val.__dict__) - 1:
                            s += '\n'
                    return s
                else:
                    return indent_str + str(val)

            return print_yaml(val)

        except Exception as ex:
            raise

    def close(self):
        """
        Bag file close.
        @param self
        """
        self.bag.close()
