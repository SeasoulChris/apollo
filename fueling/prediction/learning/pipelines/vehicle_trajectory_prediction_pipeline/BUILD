load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "vehicle_trajectory_prediction_pipeline",
    srcs = ["vehicle_trajectory_prediction_pipeline.py"],
    deps = [
        "//fueling/learning:loss_utils",
        "//fueling/learning:train_utils",
        "//fueling/prediction/learning/datasets/apollo_vehicle_trajectory_dataset",
        "//fueling/prediction/learning/models/lane_attention_trajectory_model",
        "//fueling/prediction/learning/models/semantic_map_model",
    ],
)
