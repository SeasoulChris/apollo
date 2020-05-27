#!/usr/bin/env python

import collections
import struct
import sys

from cyber.proto.record_pb2 import Channel
from cyber.proto.record_pb2 import ChunkBody
from cyber.proto.record_pb2 import ChunkHeader
from cyber.proto.record_pb2 import Index
from cyber.proto.record_pb2 import Header


PyBagMessage = collections.namedtuple('PyBagMessage',
                                      'topic message data_type timestamp')


class ApolloRecordReader:
    def __init__(self):
        self.channels = []

    def get_channels(self):
        return self.channels

    def read_messages(self, file_name, topics=None):
        with open(file_name, 'rb') as f:
            while True:
                section_type_bytes = f.read(8)
                if len(section_type_bytes) < 8:
                    print("EOF")
                    break
                section_type = struct.unpack('q', section_type_bytes)[0]

                section_len_bytes = f.read(8)
                section_len = struct.unpack('q', section_len_bytes)[0]

                if section_type == 0:
                    header = Header()
                    header.ParseFromString(f.read(2048))

                elif section_type == 4:
                    channel = Channel()
                    channel.ParseFromString(f.read(section_len))
                    self.channels.append(channel)

                elif section_type == 1:
                    chunk_header = ChunkHeader()
                    chunk_header.ParseFromString(f.read(section_len))

                elif section_type == 2:
                    chunk_body = ChunkBody()
                    chunk_body.ParseFromString(f.read(section_len))
                    for message in chunk_body.messages:
                        if topics is None:
                            yield PyBagMessage(
                                message.channel_name, message.content, "", message.time)
                        else:
                            if message.channel_name in topics:
                                yield PyBagMessage(
                                    message.channel_name, message.content, "", message.time)

                elif section_type == 3:
                    index = Index()
                    index.ParseFromString(f.read(section_len))
                    print("index size = ", len(index.indexes))
                else:
                    print("section_type = " + str(section_type))
                    print("Error: unknown section type!")
                    break


if __name__ == "__main__":
    fn = sys.argv[1]
    reader = ApolloRecordReader()
    for msg in reader.read_messages(fn):
        print(msg.topic)
        print(msg.timestamp)
