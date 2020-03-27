import struct
import sys

from cybertron_record_pb2 import ChunkHeader
from cybertron_record_pb2 import ChunkSection
from cybertron_record_pb2 import HeaderSection
from cybertron_record_pb2 import ParamSection, IndexSection, ReserveSection

from google.protobuf.descriptor_pb2 import DescriptorProto
from proto.localization_pose_pb2 import LocalizationEstimate

fn = sys.argv[1]
with open(fn, 'rb') as f:
    result = []
    cnt = 0
    while True:
        cnt += 1
        if cnt > 14:
            pass

        section_len_bytes = f.read(8)
        if len(section_len_bytes) < 8:
            print("EOF")
            break

        section_len = struct.unpack('q', section_len_bytes)[0]
        print("-------------------------------------------")
        print("len=", section_len)

        section_type_bytes = f.read(4)
        section_type = struct.unpack('i', section_type_bytes)[0]
        print("type=", section_type)

        f.read(4)

        if section_type == 7:
            header = HeaderSection()
            header.ParseFromString(f.read(2048000))
            # print(header)
            for channel in header.channels:
                print("---")
                print(channel.name)
                print(channel.type)
                # print(channel.proto_desc)
                # print(type(channel.proto_desc))

                # bytes1 = bytearray(channel.proto_desc)
                # print(type(bytes1))
                #bytes1 = channel.proto_desc.decode('utf-8')
                #DescriptorProto().ParseFromString(bytes1)
            #break
        elif section_type == 1:
            header = HeaderSection()
            header.ParseFromString(f.read(204800))


        elif section_type == 5:
            param = ParamSection()
            param.ParseFromString(f.read(section_len))

        elif section_type == 2:
            header = ChunkHeader()
            header.ParseFromString(f.read(section_len))
            print("header.rawsize = " + str(header.rawsize))

        elif section_type == 3:
            section = ChunkSection()
            section.ParseFromString(f.read(section_len))
            print("len(section.msgs) = " + str(len(section.msgs)))
            for msg in section.msgs:
                if msg.channelname == '/localization/100hz/localization_pose':
                    #print(msg.channelname)
                    localization = LocalizationEstimate()
                    localization.ParseFromString(msg.msg)
                    #print(localization.pose.position.x)
                if msg.channelname == '':
                    print(msg.channelname)

        elif section_type == 4:
            section = IndexSection()
            section.ParseFromString(f.read(section_len))
            print("len(section.indexs) = " + str(len(section.indexs)))

        elif section_type == 6:
            section = ReserveSection()
            section.ParseFromString(f.read(section_len))

        else:
            print("Error: unknown section type!")
            break
