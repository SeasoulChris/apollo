load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "multi_job_control_feature_extraction_utils",
    srcs = ["multi_job_control_feature_extraction_utils.py"],
    data = [
        "//fueling/profiling/conf:profiling_conf",
    ],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/common:sanity_check",
        "//fueling/profiling/conf:control_channel_conf",
        "//fueling/profiling/proto:control_profiling_py_pb2",
    ],
)
