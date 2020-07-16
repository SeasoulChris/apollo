load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "resource",
    srcs = glob([
        "ssl_keys/**",
        "templates/**",
    ]),
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    data = [":resource"],
    deps = [
        ":job_processor",
        "//fueling/common:mongo_utils",
    ],
)

py_library(
    name = "job_processor",
    srcs = ["job_processor.py"],
    deps = [
        ":jobs",
        ":saas_job_arg_py_pb2",
        "//fueling/common:partners",
        "//fueling/control/calibration_table:vehicle_calibration",
        "//fueling/map:generate_maps",
        "//fueling/perception/sensor_calibration:calibration_multi_sensors",
        "//fueling/profiling/control:control_profiling",
    ],
)

py_library(
    name = "jobs",
    srcs = ["jobs.py"],
    deps = [
        ":saas_job_arg_py_pb2",
        "//apps/k8s/spark_submitter:client",
        "//fueling/common:partners",
    ],
)

proto_library(
    name = "saas_job_arg_proto",
    srcs = ["saas_job_arg.proto"],
    deps = [
        "//apps/k8s/spark_submitter:spark_submit_arg_proto",
    ],
)

py_proto_library(
    name = "saas_job_arg_py_pb2",
    deps = [
        ":saas_job_arg_proto",
        "//apps/k8s/spark_submitter:spark_submit_arg_py_pb2",
    ],
)
