package(default_visibility = ["//visibility:public"])

cc_library (
  name = "py_bag",
  hdrs = ["py_bag.h"],
  srcs = ["py_bag.cc"],
  deps = [
    "//fueling/common/record/kinglong/cybertron/recorder:datafile",
    "//fueling/common/record/kinglong/cybertron/recorder:data_iterator",
    "//fueling/common/record/kinglong/cybertron/recorder:py_message_shifter",
    "//fueling/common/record/kinglong/cybertron/common:protobuf_factory",
    "//fueling/common/record/kinglong/proto/cybertron:cyber_header_proto",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_binary (
  name = "_cyber_bag.so",
  linkshared = True,
  linkstatic = False,
  srcs = ["cyber_bag_wrap.cxx"],
  deps = [
    ":py_bag",
    "//deps:python3",
  ],
)

filegroup (
  name = "cyber_bag_fg",
  srcs = [
    ":_cyber_bag.so",
  ],
)

py_binary (
  name = "cyber_bag",
  srcs = ["cyber_bag.py"],
  data = [":cyber_bag_fg"],
)

