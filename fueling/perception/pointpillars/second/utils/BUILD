load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "config_tool",
    srcs = ["config_tool.py"],
    deps = [
        "//fueling/perception/pointpillars/second/protos:pipeline_py_pb2",
        "//fueling/perception/pointpillars/second/protos:second_py_pb2",
    ],
)

py_library(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core/non_max_suppression:nms_gpu",
    ],
)

py_library(
    name = "check",
    srcs = ["check.py"],
)

py_library(
    name = "log_tool",
    srcs = ["log_tool.py"],
)

py_library(
    name = "progress_bar",
    srcs = ["progress_bar.py"],
)

py_library(
    name = "timer",
    srcs = ["timer.py"],
)

py_library(
    name = "simplevis",
    srcs = ["simplevis.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
    ],
)

py_library(
    name = "find",
    srcs = ["find.py"],
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
)
