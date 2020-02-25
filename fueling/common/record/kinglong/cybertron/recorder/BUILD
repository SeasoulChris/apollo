package(default_visibility = ["//visibility:public"])

cc_library(
    name = "py_message_shifter",
    hdrs = ["py_message_shifter.h"],
    deps = [
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/cybertron/common:message_base",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "param",
    hdrs = ["param.h"],
    deps = [
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "chunk",
    srcs = ["chunk.cc"],
    hdrs = ["chunk.h"],
    deps = [
        ":param",
        "//fueling/common/record/kinglong/cybertron/common:define",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "compress",
    srcs = ["compress.cc"],
    hdrs = ["compress.h"],
    deps = [
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "filter",
    srcs = ["filter.cc"],
    hdrs = ["filter.h"],
    deps = [
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "fileopt",
    srcs = ["fileopt.cc"],
    hdrs = ["fileopt.h"],
    deps = [
        ":compress",
        "//fueling/common/record/kinglong/cybertron/common:define",
        "//fueling/common/record/kinglong/cybertron/common:file_util",
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/cybertron/common:md5",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "datafile",
    srcs = ["datafile.cc"],
    hdrs = ["datafile.h"],
    deps = [
        ":chunk",
        ":fileopt",
        ":param",
        "//fueling/common/record/kinglong/cybertron/common:conf_manager",
        "//fueling/common/record/kinglong/cybertron/common:define",
        "//fueling/common/record/kinglong/cybertron/common:file_util",
        "//fueling/common/record/kinglong/cybertron/common:macros",
        "//fueling/common/record/kinglong/cybertron/common:md5",
        "//fueling/common/record/kinglong/cybertron/common:protobuf_factory",
        "//fueling/common/record/kinglong/cybertron/common:raw_message",
        "//fueling/common/record/kinglong/proto/cybertron:parameter_proto",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)

cc_library(
    name = "data_iterator",
    srcs = ["data_iterator.cc"],
    hdrs = ["data_iterator.h"],
    deps = [
        ":datafile",
        ":filter",
        "//fueling/common/record/kinglong/proto/cybertron:parameter_proto",
        "//fueling/common/record/kinglong/proto/cybertron:record_proto",
    ],
)
