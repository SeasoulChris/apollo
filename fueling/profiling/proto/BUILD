load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "control_profiling_data_proto",
    srcs = ["control_profiling_data.proto"],
)

py_proto_library(
    name = "control_profiling_data_py_pb2",
    deps = [
        ":control_profiling_data_proto",
    ],
)

proto_library(
    name = "control_profiling_proto",
    srcs = ["control_profiling.proto"],
)

py_proto_library(
    name = "control_profiling_py_pb2",
    deps = [
        ":control_profiling_proto",
    ],
)

proto_library(
    name = "open_space_planner_profiling_proto",
    srcs = ["open_space_planner_profiling.proto"],
)

py_proto_library(
    name = "open_space_planner_profiling_py_pb2",
    deps = [
        ":open_space_planner_profiling_proto",
    ],
)
