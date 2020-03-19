package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        "//fueling/common:logging",
        requirement("pyro-ppl"),
    ],
)