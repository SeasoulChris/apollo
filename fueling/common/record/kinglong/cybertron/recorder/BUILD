package(default_visibility = ["//visibility:public"])

cc_library (
  name = "py_message_shifter",
  hdrs = ["py_message_shifter.h"],
  deps = [
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/cybertron/common:message_base",
  ],
)

cc_library (
  name = "param",
  hdrs = ["param.h"],
  deps = [
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_library (
  name = "chunk",
  hdrs = ["chunk.h"],
  deps = [
    ":param",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_library (
  name = "compress",
  hdrs = ["compress.h"],
  srcs = ["compress.cc"],
  deps = [
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_library (
  name = "filter",
  hdrs = ["filter.h"],
  srcs = ["filter.cc"],
  deps = [
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_library (
  name = "fileopt",
  hdrs = ["fileopt.h"],
  srcs = ["fileopt.cc"],
  deps = [
    ":compress",
    "//fueling/common/record/kinglong/cybertron/common:define",
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/cybertron/common:md5",
    "//fueling/common/record/kinglong/cybertron/common:file_util",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
  ],
)

cc_library (
  name = "datafile",
  hdrs = ["datafile.h"],
  srcs = ["datafile.cc"],
  deps = [
    ":chunk",
    ":fileopt",
    ":param",
    "//fueling/common/record/kinglong/cybertron/common:define",
    "//fueling/common/record/kinglong/cybertron/common:macros",
    "//fueling/common/record/kinglong/cybertron/common:md5",
    "//fueling/common/record/kinglong/cybertron/common:file_util",
    "//fueling/common/record/kinglong/cybertron/common:protobuf_factory",
    "//fueling/common/record/kinglong/cybertron/common:raw_message",
    "//fueling/common/record/kinglong/cybertron/common:conf_manager",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    "//fueling/common/record/kinglong/proto/cybertron:parameter_proto",
  ],
)

cc_library (
  name = "data_iterator",
  hdrs = ["data_iterator.h"],
  srcs = ["data_iterator.cc"],
  deps = [
    ":datafile",
    ":filter",
    "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    "//fueling/common/record/kinglong/proto/cybertron:parameter_proto",
  ],
)
