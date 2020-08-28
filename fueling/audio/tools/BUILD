load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "direction_detection",
    srcs = ["direction_detection.py"],
    deps = [],
)

py_binary(
    name = "auto_label",
    srcs = ["auto_label.py"],
    deps = [
        ":direction_detection",
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
        "@apollo//modules/common/proto:header_py_pb2",
        "@apollo//modules/drivers/microphone/proto:audio_py_pb2",
        "@apollo//modules/drivers/microphone/proto:microphone_config_py_pb2",
    ],
)

py_binary(
    name = "audio_saver",
    srcs = ["audio_saver.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
        "@apollo//modules/drivers/microphone/proto:microphone_config_py_pb2",
    ],
)
