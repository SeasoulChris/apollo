load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "create_data",
    srcs = ["create_data.py"],
    deps = [
        "//fueling/perception/pointpillars/second/data:all_dataset",
        "//fueling/perception/pointpillars/second/data:kitti_dataset",
        "//fueling/perception/pointpillars/second/data:nuscenes_dataset",
    ],
)
