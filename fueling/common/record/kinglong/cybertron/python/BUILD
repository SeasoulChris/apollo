package(default_visibility = ["//visibility:public"])

cc_library(
    name = "py_bag",
    srcs = ["py_bag.cc"],
    hdrs = ["py_bag.h"],
    deps = [
        "//fueling/common/record/kinglong/cybertron/common:protobuf_factory",
        "//fueling/common/record/kinglong/cybertron/recorder:data_iterator",
        "//fueling/common/record/kinglong/cybertron/recorder:datafile",
        "//fueling/common/record/kinglong/cybertron/recorder:py_message_shifter",
        "//fueling/common/record/kinglong/proto/cybertron:cyber_header_proto",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_binary(
    name = "_cyber_bag.so",
    srcs = ["cyber_bag_wrap.cxx"],
    linkshared = True,
    linkstatic = False,
    deps = [
        ":py_bag",
        "//deps:python3",
    ],
)

py_library(
    name = "cyber_bag",
    srcs = ["cyber_bag.py"],
    data = [":_cyber_bag.so"],
)

py_library(
    name = "bag",
    srcs = ["bag.py"],
    deps = [":cyber_bag"],
)

py_binary(
    name = "convert",
    srcs = ["convert.py"],
    deps = [
        ":bag",
        "//fueling/common/record/kinglong/proto/modules:localization_pose_pb2",
        "//fueling/common/record/kinglong/proto/modules:perception_obstacle_pb2",
    ],
)
