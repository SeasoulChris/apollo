package(default_visibility = ["//visibility:public"])

py_binary(
    name = "semantic_map_pipeline",
    srcs = ["semantic_map_pipeline.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/planning/datasets:semantic_map_dataset",
        "//fueling/planning/models:semantic_map_model",
    ],
)

py_binary(
    name = "model_training_on_cloud",
    srcs = [
        "model_training_on_cloud.py",
        "trajectory_imitation_model_training_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/learning:train_utils",
        "//fueling/planning/datasets:img_in_traj_out_dataset",
        "//fueling/planning/models:trajectory_imitation_model",
    ],
)
