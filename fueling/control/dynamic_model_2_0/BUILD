package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/gp_regression/pyro_archive:evaluation",
        "//fueling/control/dynamic_model_2_0/gp_regression/pyro_archive:dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression/pyro_archive:gaussian_process",
        "//fueling/control/dynamic_model_2_0/gp_regression/pyro_archive:train",
    ],
)
