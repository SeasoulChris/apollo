package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":model_conf",
        "//fueling/common:logging",
    ],
)

py_library(
    name = "deep_encoding_net",
    srcs = ["deep_encoding_net.py"],
)

py_library(
    name = "dreamview_server",
    srcs = ["dreamview_server.py"],
    deps = [
        requirement("pyro-ppl"),
        ":train",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":dataset",
        ":train",
        "//fueling/common:logging",
    ],
)

py_library(
    name = "gaussian_process",
    srcs = ["gaussian_process.py"],
    deps = [
        requirement("pyro-ppl"),
    ],
)

py_binary(
    name = "label_generation",
    srcs = ["label_generation.py"],
    deps = [
        ":model_conf",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":dataset",
        ":dreamview_server",
        ":evaluation",
        ":gaussian_process",
        ":train",
    ],
)

py_library(
    name = "model_conf",
    srcs = ["model_conf.py"],
)

py_library(
    name = "predict",
    srcs = ["predict.py"],
    deps = [
        requirement("pyro-ppl"),
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":deep_encoding_net",
        ":model_conf",
        ":predict",
        requirement("graphviz"),
        requirement("tqdm"),
        requirement("opt-einsum"),
        requirement("progressbar"),
        requirement("pyro-ppl"),
    ],
)
