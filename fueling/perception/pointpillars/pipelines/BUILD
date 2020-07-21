load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "pointpillars_training_pipeline",
    srcs = [
        "pointpillars_training_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/perception/pointpillars/second/pytorch:train",
    ],
)

py_binary(
    name = "create_data_pipeline",
    srcs = [
        "create_data_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/perception/pointpillars/second:create_data",
    ],
)
