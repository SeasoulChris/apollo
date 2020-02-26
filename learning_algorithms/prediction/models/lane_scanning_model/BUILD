package(default_visibility = ["//visibility:public"])

py_library(
    name = "lane_scanning_model",
    srcs = ["lane_scanning_model.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/learning:loss_utils",
        "//fueling/learning:network_utils",
    ],
)

py_library(
    name = "pipeline_lane_scanning",
    srcs = ["pipeline_lane_scanning.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/learning:train_utils",
        "//learning_algorithms/prediction/models/lane_scanning_model",
    ],
)
