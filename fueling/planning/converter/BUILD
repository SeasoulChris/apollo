load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "data_converter",
    srcs = ["data_converter.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "@apollo//cyber/python/cyber_py3:record",
        "@planning_analytics",
    ],
)
