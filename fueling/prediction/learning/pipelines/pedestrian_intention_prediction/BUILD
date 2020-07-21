load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pedestrian_intention_dataset",
    srcs = ["pedestrian_intention_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ],
)

py_library(
    name = "key_pedestrian_model",
    srcs = ["key_pedestrian_model.py"],
)
