package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [
        requirement("pyro-ppl"),
    ],
)

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

py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
    deps = [
        requirement("pyro-ppl"),
    ],
)

py_library(
    name = "gp_model",
    srcs = ["gp_model.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        requirement("gpytorch"),
    ],
)

py_library(
    name = "gp_model_example",
    srcs = ["gp_model_example.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        requirement("gpytorch"),
    ],
)

py_binary(
    name = "gpytorch_train_example",
    srcs = ["gpytorch_train_example.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/gp_regression:dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        "//fueling/control/dynamic_model_2_0/gp_regression:gp_model_example",
    ],
)

py_binary(
    name = "gpytorch_train",
    srcs = ["gpytorch_train.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/gp_regression:dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        "//fueling/control/dynamic_model_2_0/gp_regression:gp_model",
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":encoder",
        ":predict",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        requirement("pyro-ppl"),
    ],
)
