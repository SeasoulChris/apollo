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
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":encoder",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        ":predict",
        requirement("graphviz"),
        requirement("tqdm"),
        requirement("opt-einsum"),
        requirement("progressbar"),
        requirement("pyro-ppl"),
    ],
)
