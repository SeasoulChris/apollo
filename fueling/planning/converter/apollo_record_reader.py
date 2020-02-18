import sys
import struct
from cyber.proto.record_pb2 import SectionType
from cyber.proto.record_pb2 import Header
from cyber.proto.record_pb2 import Channel
from cyber.proto.record_pb2 import ChunkHeader
from cyber.proto.record_pb2 import ChunkBody
from cyber.proto.record_pb2 import Index

fn = sys.argv[1]
with open(fn, 'rb') as f:
    result = []
    cnt = 0
    while True:
        section_type_bytes = f.read(8)
        if len(section_type_bytes) < 8:
            print("EOF")
            break
        section_type = struct.unpack('q', section_type_bytes)[0]
        print("-------------------------------------------")
        print("type=", section_type)

        section_len_bytes = f.read(8)
        section_len = struct.unpack('q', section_len_bytes)[0]
        print("len=", section_len)

        if section_type == 0:
            header = Header()
            header.ParseFromString(f.read(2048))
            print(header)
        elif section_type == 4:
            channel = Channel()
            channel.ParseFromString(f.read(section_len))
            print(channel.name)
        elif section_type == 1:
            chunk_header = ChunkHeader()
            chunk_header.ParseFromString(f.read(section_len))
            print("chunk header raw size = ", chunk_header.raw_size)
            print("chunk header message_number = ", chunk_header.message_number)
        elif section_type == 2:
            chunk_body = ChunkBody()
            chunk_body.ParseFromString(f.read(section_len))
            print("chunk body msg size = ", len(chunk_body.messages))
        elif section_type == 3:
            index = Index()
            index.ParseFromString(f.read(section_len))
            print("index size = ", len(index.indexes))
        else:
            print("Error: unknown section type!")
