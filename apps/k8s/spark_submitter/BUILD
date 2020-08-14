load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "spark_submit_arg_proto",
    srcs = ["spark_submit_arg.proto"],
)

py_proto_library(
    name = "spark_submit_arg_py_pb2",
    deps = [
        ":spark_submit_arg_proto",
    ],
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    deps = [
        ":job_processor",
        ":utils",
    ],
)

py_library(
    name = "client",
    srcs = ["client.py"],
    deps = [
        ":utils",
        "//fueling/common:file_utils",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":spark_submit_arg_py_pb2",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:proto_utils",
    ],
)

py_library(
    name = "job_processor",
    srcs = ["job_processor.py"],
    deps = [
        ":jobs",
        ":saas_job_arg_py_pb2",
        "//fueling/control/calibration_table:vehicle_calibration",
        "//fueling/map:generate_maps",
        "//fueling/perception/sensor_calibration:calibration_multi_sensors",
        "//fueling/profiling/control:control_profiling",
        "//fueling/profiling/open_space_planner:metrics",
    ],
)

py_library(
    name = "jobs",
    srcs = ["jobs.py"],
    deps = [
        ":client",
    ],
)

proto_library(
    name = "saas_job_arg_proto",
    srcs = ["saas_job_arg.proto"],
    deps = [
        ":spark_submit_arg_proto",
    ],
)

py_proto_library(
    name = "saas_job_arg_py_pb2",
    deps = [
        ":saas_job_arg_proto",
        ":spark_submit_arg_py_pb2",
    ],
)
