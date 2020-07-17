package(default_visibility = ["//visibility:public"])

py_library(
    name = "kitti_common",
    srcs = ["kitti_common.py"],

)

py_library(
    name = "all_dataset",
    srcs = ["all_dataset.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
        ":kitti_dataset",
        ":nuscenes_dataset",
        ":dataset",
    ],
)

py_library(
    name = "kitti_dataset",
    srcs = ["kitti_dataset.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
        "//fueling/perception/pointpillars/second/utils:eval",
        ":kitti_common",
        ":dataset",
    ],
)

py_library(
    name = "nuscenes_dataset",
    srcs = ["nuscenes_dataset.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
        "//fueling/perception/pointpillars/second/utils:eval",
        "//fueling/perception/pointpillars/second/utils:timer",
        ":kitti_common",
        ":dataset",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        ":kitti_common",
    ],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        "//fueling/perception/pointpillars/second/core:geometry",
        ":kitti_common",
        "//fueling/perception/pointpillars/second/utils:simplevis",
        "//fueling/perception/pointpillars/second/utils:timer",
    ],
)
