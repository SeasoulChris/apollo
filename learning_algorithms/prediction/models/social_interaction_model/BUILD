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
        "//learning_algorithms/prediction/datasets/apollo_pedestrian_dataset",
        "//learning_algorithms/prediction/models/social_interaction_model",
        "//learning_algorithms/prediction/models/social_interaction_model:human_trajectory_dataset",
    ],
)
