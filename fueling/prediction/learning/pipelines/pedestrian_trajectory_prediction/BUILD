package(default_visibility = ["//visibility:public"])

py_library(
    name = "pedestrian_trajectory_dataset",
    srcs = ["pedestrian_trajectory_dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ]
)

py_binary(
    name = "pedestrian_trajectory_pipeline",
    srcs = ["pedestrian_trajectory_pipeline.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning:loss_utils",
        "//fueling/learning:train_utils",
        "//fueling/prediction/common:configure",
        ":pedestrian_trajectory_dataset",
        "//fueling/prediction/learning/models/semantic_map_model:semantic_map_model",
    ]
)

py_binary(
    name = "visualize_model",
    srcs = ["visualize_model.py"],
    deps = [
        ":pedestrian_trajectory_dataset",
        "//fueling/prediction/learning/models/semantic_map_model:semantic_map_model",
    ],
)
