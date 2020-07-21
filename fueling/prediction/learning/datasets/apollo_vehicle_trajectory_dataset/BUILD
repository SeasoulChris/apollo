load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "apollo_vehicle_trajectory_dataset",
    srcs = ["apollo_vehicle_trajectory_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "apollo_vehicle_trajectory_cleaning",
    srcs = ["apollo_vehicle_trajectory_cleaning.py"],
)

py_binary(
    name = "test_apollo_vehicle_trajectory_dataset",
    srcs = ["test_apollo_vehicle_trajectory_dataset.py"],
    deps = [
        ":apollo_vehicle_trajectory_dataset",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)

py_binary(
    name = "visualize_model",
    srcs = ["visualize_model.py"],
    deps = [
        ":apollo_vehicle_trajectory_dataset",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)
