package(default_visibility = ["//visibility:public"])

py_library(
    name = "semantic_map_model",
    srcs = ["semantic_map_model.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/learning:network_utils",
        "//fueling/learning:train_utils",
    ],
)

py_binary(
    name = "semantic_map_pipeline",
    srcs = ["semantic_map_pipeline.py"],
    deps = [
        ":semantic_map_model",
        "//fueling/learning:train_utils",
    ],
)