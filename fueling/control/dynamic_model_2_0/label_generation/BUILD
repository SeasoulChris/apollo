package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "label_generation",
    srcs = ["label_generation.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)
