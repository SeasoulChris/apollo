package(default_visibility = ["//visibility:public"])

py_library(
    name = "coord_conversion_utils",
    srcs = ["coord_conversion_utils.py"],
    deps = [
        "//fueling/learning:network_utils",
        "//fueling/learning:train_utils",
    ],
)

py_library(
    name = "lane_attention_trajectory_model",
    srcs = ["lane_attention_trajectory_model.py"],
    deps = [
        "//fueling/learning:network_utils",
        "//fueling/learning:train_utils",
        "//learning_algorithms/prediction/models/lane_attention_trajectory_model:coord_conversion_utils",
    ],
)
