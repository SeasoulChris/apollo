package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "feature_extraction_utils",
    srcs = ["feature_extraction_utils.py"],
    deps = [
        requirement("shapely"),
        "//fueling/common:logging",
        "//fueling/profiling/common:multi_vehicle_utils",
    ],
)
