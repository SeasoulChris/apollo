load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "echo_lincoln",
    srcs = ["echo_lincoln.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model/data_generator:feature_extraction",
    ],
)
