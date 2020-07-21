load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "feature_extraction_utils",
    srcs = ["feature_extraction_utils.py"],
    deps = [
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
        "//fueling/profiling/common:multi_vehicle_utils",
    ],
)

py_library(
    name = "feature_visualization_utils",
    srcs = ["feature_visualization_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/profiling/conf:open_space_planner_conf",
    ],
)
