load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "semantic_map_model",
    srcs = ["semantic_map_model.py"],
    deps = [
        "//fueling/common:coord_utils",
    ],
)

py_library(
    name = "trajectory_imitation_model",
    srcs = ["trajectory_imitation_model.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:logging",
        "//fueling/learning:network_utils",
    ],
)
