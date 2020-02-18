package(default_visibility = ["//visibility:public"])

py_library(
    name = "latency",
    srcs = ["latency.py"],
)

py_library(
    name = "lidar_endtoend_analyzer",
    srcs = ["lidar_endtoend_analyzer.py"],
    deps = [
        "//fueling/planning/metrics/common:statistical_analyzer",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":module_control_analyzer",
        ":module_planning_analyzer",
        "//fueling/common:record_utils",
        "//fueling/planning/metrics:lidar_endtoend_analyzer",
        "//fueling/planning/stability/grading:planning_stability_grader",
    ],
)

py_library(
    name = "module_control_analyzer",
    srcs = ["module_control_analyzer.py"],
    deps = [
        "//fueling/planning/metrics/common:error_code_analyzer",
        "//fueling/planning/metrics/common:error_msg_analyzer",
        "//fueling/planning/metrics/common:statistical_analyzer",
    ],
)

py_library(
    name = "module_planning_analyzer",
    srcs = ["module_planning_analyzer.py"],
    deps = [
        "//fueling/planning/metrics/common:distribution_analyzer",
        "//fueling/planning/metrics/common:error_code_analyzer",
        "//fueling/planning/metrics/common:error_msg_analyzer",
        "//fueling/planning/metrics/common:statistical_analyzer",
        "//fueling/planning/metrics/metrics:curvature",
        "//fueling/planning/metrics/metrics:frame_count",
        "//fueling/planning/metrics/metrics:lat_acceleration",
        "//fueling/planning/metrics/metrics:lon_acceleration",
        "//fueling/planning/metrics/metrics:latency",
        "//fueling/planning/metrics/metrics:reference_line",
    ],
)
