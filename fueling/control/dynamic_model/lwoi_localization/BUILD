package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        requirement("liegroups"),
    ]
)

py_binary(
    name = "downloader",
    srcs = ["downloader.py"],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    deps = [
        requirement("liegroups"),
    ]
)

py_library(
    name = "main_lwoi",
    srcs = ["main_lwoi.py"],
    deps = [
        ":dataset",
        ":filter",
        ":train",
    ]
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":utils",
    ]
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        requirement("pyro-ppl"),
    ]
)
