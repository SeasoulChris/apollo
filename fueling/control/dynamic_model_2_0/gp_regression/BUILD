package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
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
    name = "gpytorch_train",
    srcs = ["gpytorch_train.py"],
    deps = [
        ":dataset",
        ":gp_model",
    ],
)

py_library(
    name = "gp_model",
    srcs = ["gp_model.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        ":encoder",
        requirement("gpytorch"),
    ],
)
