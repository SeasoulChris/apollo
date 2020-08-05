load("@rules_python//python:defs.bzl", "py_binary")

filegroup(
    name = "task_list",
    srcs = ["task_list.txt"],
)

py_binary(
    name = "routing_generator",
    srcs = ["routing_generator.py"],
    data = [
        "//fueling/planning/routing_generator:task_list",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "@apollo//cyber/python/cyber_py3:record",
        "@planning_analytics",
    ],
)
