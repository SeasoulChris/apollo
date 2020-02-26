package(default_visibility = ["//visibility:public"])

py_library(
    name = "test_planning_trajectory_dataset",
    srcs = ["test_planning_trajectory_dataset.py"],
)

py_library(
    name = "planning_trajectory_cleaning",
    srcs = ["planning_trajectory_cleaning.py"],
)

py_library(
    name = "planning_trajectory_dataset",
    srcs = ["planning_trajectory_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
    ],
)
