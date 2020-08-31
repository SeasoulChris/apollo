load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "audio_detection_profiling",
    srcs = ["audio_detection_profiling.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/audio/proto:audio_common_py_pb2",
        "@apollo//modules/audio/proto:audio_event_py_pb2",
        "@apollo//modules/audio/proto:audio_py_pb2",
    ],
)
