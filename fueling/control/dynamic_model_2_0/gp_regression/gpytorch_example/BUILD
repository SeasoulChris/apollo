load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "gp_model_example",
    srcs = ["gp_model_example.py"],
    deps = [
        "//fueling/common:logging",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_binary(
    name = "gpytorch_train_example",
    srcs = ["gpytorch_train_example.py"],
    deps = [
        ":dataset",
        ":gp_model_example",
    ],
)
