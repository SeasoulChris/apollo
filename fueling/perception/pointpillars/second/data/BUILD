load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kitti_common",
    srcs = ["kitti_common.py"],
)

py_library(
    name = "all_dataset",
    srcs = ["all_dataset.py"],
    deps = [
        ":dataset",
        ":kitti_dataset",
        ":nuscenes_dataset",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
    ],
)

py_library(
    name = "kitti_dataset",
    srcs = ["kitti_dataset.py"],
    deps = [
        ":dataset",
        ":kitti_common",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/utils:eval",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
    ],
)

py_library(
    name = "nusc_eval",
    srcs = ["nusc_eval.py"],
)

py_library(
    name = "nuscenes_dataset",
    srcs = ["nuscenes_dataset.py"],
    deps = [
        ":dataset",
        ":kitti_common",
        ":nusc_eval",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/utils:eval",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
        "//fueling/perception/pointpillars/second/utils:timer",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":kitti_common",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
    ],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        ":kitti_common",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:geometry",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/utils:simplevis",
        "//fueling/perception/pointpillars/second/utils:timer",
    ],
)
