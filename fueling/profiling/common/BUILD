package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "multi_vehicle_utils",
    srcs = ["multi_vehicle_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
    ]
)

py_library(
    name = "dir_utils",
    srcs = ["dir_utils.py"],
    deps = [
        ":multi_vehicle_utils",
        "//fueling/common:base_pipeline_v2",
        "//fueling/common/storage:bos_client",
    ]
)

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        ":multi_vehicle_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:email_utils",
        "//fueling/common:record_utils",
        
    ]
)
