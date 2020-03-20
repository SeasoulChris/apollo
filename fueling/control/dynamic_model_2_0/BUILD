package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/evaluation",
        "//fueling/control/dynamic_model_2_0/gp_regression:dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression:gaussian_process",
        "//fueling/control/dynamic_model_2_0/gp_regression:train",
    ],
)
