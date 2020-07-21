load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "human_trajectory_dataset",
    srcs = ["human_trajectory_dataset.py"],
)

py_library(
    name = "social_interaction_model",
    srcs = ["social_interaction_model.py"],
    deps = [
        "//fueling/learning:train_utils",
    ],
)

py_library(
    name = "pipeline_social_interaction",
    srcs = ["pipeline_social_interaction.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/prediction/learning/datasets/apollo_pedestrian_dataset",
        "//fueling/prediction/learning/models/social_interaction_model",
        "//fueling/prediction/learning/models/social_interaction_model:human_trajectory_dataset",
    ],
)
