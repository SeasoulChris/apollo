package(default_visibility = ["//visibility:public"])

load("//tools:py_proto.bzl", "py_proto")

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
        ":saas_job_arg_pb2",
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
        "//apps/k8s/spark_submitter:client",
    ],
)

py_proto(
    name = "saas_job_arg_pb2",
    src = "saas_job_arg.proto",
    deps = [
        "//apps/k8s/spark_submitter:spark_submit_arg_pb2",
    ],
)
