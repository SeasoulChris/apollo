package(default_visibility = ["//visibility:public"])

py_library(
    name = "junction_map_model",
    srcs = ["junction_map_model.py"],
    deps = [
        "//fueling/common:coord_utils",
    ],
)

py_library(
    name = "img_test",
    srcs = ["img_test.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//learning_algorithms/prediction/models/junction_map_model",
    ],
)

py_library(
    name = "junction_map_pipeline",
    srcs = ["junction_map_pipeline.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//learning_algorithms/prediction/models/junction_map_model",
    ],
)
