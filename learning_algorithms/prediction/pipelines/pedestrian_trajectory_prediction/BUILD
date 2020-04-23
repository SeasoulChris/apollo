package(default_visibility = ["//visibility:public"])

py_library(
    name = "pedestrian_trajectory_dataset",
    srcs = ["pedestrian_trajectory_dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//learning_algorithms/prediction/data_preprocessing/map_feature:online_mapping",
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
        "//learning_algorithms/prediction/models/semantic_map_model:semantic_map_model",
    ]
)
