package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "control_profiling_metrics",
    srcs = ["multi_job_control_profiling_metrics.py"],
    deps = [
        "//fueling/common:base_pipeline_v2",
        "//fueling/common:file_utils//fueling/common:email_utils",
        "//fueling/common:logging//fueling/common:partners",
    ],
)

py_binary(
    name = "control_profiling",
    srcs = ["control_profiling.py"],
    deps = [
        ":control_profiling_metrics",
        ":control_profiling_visualization",
    ],
)
