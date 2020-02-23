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
