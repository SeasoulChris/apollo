package(default_visibility = ["//visibility:public"])

py_library(
    name = "apollo_vehicle_trajectory_dataset",
    srcs = ["apollo_vehicle_trajectory_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//learning_algorithms/prediction/data_preprocessing/map_feature:online_mapping",
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
        "//learning_algorithms/prediction/models/semantic_map_model:semantic_map_model",
    ],
)
