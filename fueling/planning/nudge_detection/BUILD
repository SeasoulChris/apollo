load("@rules_python//python:defs.bzl", "py_binary")

filegroup(
    name = "task_list",
    srcs = ["task_list.txt"],
)

py_binary(
    name = "nudge_detection",
    srcs = ["nudge_detection.py"],
    data = [
        "//fueling/planning/nudge_detection:task_list",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "@planning_analytics",
    ],
)
