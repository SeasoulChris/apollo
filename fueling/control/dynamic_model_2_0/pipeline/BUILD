load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "data_splitter",
    srcs = ["data_splitter.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
    ],
)
