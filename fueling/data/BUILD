load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "record_parser",
    srcs = ["record_parser.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:record_utils",
        "//fueling/data/proto:record_meta_py_pb2",
        "//fueling/planning/metrics:latency",
        "//fueling/planning/stability:speed_jerk_stability",
        "//fueling/planning/stability/grading:imu_stability_grader",
        "//fueling/planning/stability/grading:planning_stability_grader",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/localization/proto:gps_py_pb2",
        "@apollo//modules/localization/proto:localization_py_pb2",
        "@apollo//modules/planning/proto:planning_py_pb2",
    ],
)
