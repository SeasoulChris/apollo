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
        ":client",
        ":spark_submit_arg_py_pb2",
        "//apps/web_portal:jobs",
        "//apps/web_portal:saas_job_arg_py_pb2",
        "//fueling/common:job_utils",
        "//fueling/common:mongo_utils",
        "//fueling/common:proto_utils",
        "//fueling/control/calibration_table:vehicle_calibration",
        "//fueling/map:generate_maps",
        "//fueling/perception/sensor_calibration:calibration_multi_sensors",
        "//fueling/profiling/control:control_profiling",
    ],
)

py_library(
    name = "client",
    srcs = ["client.py"],
    deps = [
        "//fueling/common:file_utils",
    ],
)
