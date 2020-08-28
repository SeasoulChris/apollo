load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pedestrian_trajectory_dataset",
    srcs = ["pedestrian_trajectory_dataset.py"],
    data = [
        "//testdata/map_feature:prediction_semantic_map",
    ],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ],
)

py_library(
    name = "pedestrian_trajectory_dataset_cloud",
    srcs = ["pedestrian_trajectory_dataset_cloud.py"],
    data = [
        "//testdata/map_feature:prediction_semantic_map",
    ],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/prediction/learning/data_preprocessing/map_feature:mapping",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ],
)

py_binary(
    name = "pedestrian_trajectory_pipeline",
    srcs = ["pedestrian_trajectory_pipeline.py"],
    deps = [
        ":pedestrian_trajectory_dataset",
        "//fueling/common:logging",
        "//fueling/learning:loss_utils",
        "//fueling/learning:train_utils",
        "//fueling/prediction/common:configure",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)

py_binary(
    name = "visualize_model",
    srcs = ["visualize_model.py"],
    deps = [
        ":pedestrian_trajectory_dataset",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)

py_binary(
    name = "pedestrian_trajectory_training",
    srcs = ["pedestrian_trajectory_training.py"],
    data = [
        "//testdata/prediction:model_files",
    ],
    deps = [
        ":pedestrian_trajectory_dataset_cloud",
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
        "//fueling/learning:loss_utils",
        "//fueling/learning:train_utils",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)
