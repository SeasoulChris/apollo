package(default_visibility = ["//visibility:public"])

py_binary(
    name = "semantic_map_pipeline",
    srcs = ["semantic_map_pipeline.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/planning/datasets:semantic_map_dataset",
        "//fueling/planning/models:semantic_map_model",
    ]
)
