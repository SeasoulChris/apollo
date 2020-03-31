package(default_visibility = ["//visibility:public"])

py_binary(
    name = "pedestrian_trajectory_dataset",
    srcs = ["pedestrian_trajectory_dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//learning_algorithms/prediction/data_preprocessing/map_feature:online_mapping",
    ]
)
