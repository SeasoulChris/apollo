load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "junction_mlp_pipeline",
    srcs = ["junction_mlp_pipeline.py"],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/prediction/learning/models/junction_mlp_model",
    ],
)

py_library(
    name = "junction_mlp_model",
    srcs = ["junction_mlp_model.py"],
)
