load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "multi_vehicle_utils",
    srcs = ["multi_vehicle_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
    ],
)

py_library(
    name = "dir_utils",
    srcs = ["dir_utils.py"],
    deps = [
        ":multi_vehicle_utils",
        "//fueling/common:base_pipeline",
        "//fueling/common/storage:bos_client",
    ],
)

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        ":multi_vehicle_utils",
        "//fueling/common:email_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
    ],
)

py_library(
    name = "stats_utils",
    srcs = ["stats_utils.py"],
    deps = [
        "//fueling/common:logging",
    ],
)
