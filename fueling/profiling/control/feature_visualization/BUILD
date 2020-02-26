package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "control_feature_visualization_utils",
    srcs = ["control_feature_visualization_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:json_utils",
        "//fueling/profiling/proto:control_profiling",
        "//fueling/profiling/conf:control_channel_conf",
        "//fueling/profiling/proto:control_profiling_data",
    ]
)