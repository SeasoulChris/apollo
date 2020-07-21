load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "predict",
    srcs = ["predict.py"],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    deps = [
        "//fueling/common:logging",
    ],
)

py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":encoder",
        ":predict",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":encoder",
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
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":dataset",
        ":evaluation",
        ":gaussian_process",
        ":train",
    ],
)
