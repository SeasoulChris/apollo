package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "raw_data_visualization",
    srcs = ["raw_data_visualization.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/label_generation",
    ],
)
