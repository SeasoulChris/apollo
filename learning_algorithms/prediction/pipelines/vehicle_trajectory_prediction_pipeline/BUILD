package(default_visibility = ["//visibility:public"])

py_binary(
    name = "vehicle_trajectory_prediction_pipeline",
    srcs = ["vehicle_trajectory_prediction_pipeline.py"],
    deps = [
        "//fueling/learning:loss_utils",
        "//fueling/learning:train_utils",
        "//learning_algorithms/prediction/datasets/apollo_vehicle_trajectory_dataset",
        "//learning_algorithms/prediction/models/lane_attention_trajectory_model",
        "//learning_algorithms/prediction/models/semantic_map_model",
    ],
)
