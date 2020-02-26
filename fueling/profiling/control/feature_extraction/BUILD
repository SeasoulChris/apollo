package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "multi_job_control_feature_extraction_utils",
    srcs = ["multi_job_control_feature_extraction_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:sanity_check",
        "//fueling/profiling/common:multi_vehicle_utils",
	"//fueling/profiling/conf:control_channel_conf",
        "//fueling/profiling/proto:control_profiling",
    ]
)


