package(default_visibility = ["//visibility:public"])

py_library(
    name = "control_profiling_metrics",
    srcs = ["control_profiling_metrics.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
    ],
)

py_library(
    name = "control_profiling_visualization",
    srcs = ["control_profiling_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
    ],
)

py_binary(
    name = "control_profiling",
    srcs = ["control_profiling.py"],
    deps = [
        ":multi_job_control_profiling_metrics",
        ":multi_job_control_profiling_visualization",
        "//fueling/common:base_pipeline",
    ],
)

py_library(
    name = "control_profiling_scenario_oriented_visualization",
    srcs = ["control_profiling_scenario_oriented_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/control/feature_visualization:control_feature_scenario_oriented_visualization_utils",
    ],
)

py_library(
    name = "multi_job_control_profiling_visualization_test",
    srcs = ["multi_job_control_profiling_visualization_test.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/control/feature_visualization:control_feature_visualization_utils",
    ],
)

py_library(
    name = "reorg_smallrecords_by_vehicle",
    srcs = ["reorg_smallrecords_by_vehicle.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/control/feature_extraction:multi_job_control_feature_extraction_utils",
    ],
)

py_library(
    name = "multi_job_control_profiling_metrics_test",
    srcs = ["multi_job_control_profiling_metrics_test.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/common:sanity_check",
        "//fueling/profiling/control:multi_job_control_profiling_metrics",
        "//fueling/profiling/control/feature_extraction:multi_job_control_feature_extraction_utils",
        "//fueling/profiling/control/grading_evaluation:multi_job_control_performance_grading_utils",
    ],
)

py_library(
    name = "multi_job_control_profiling_visualization",
    srcs = ["multi_job_control_profiling_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/control/feature_visualization:control_feature_visualization_utils",
    ],
)

py_library(
    name = "multi_job_control_profiling_metrics",
    srcs = ["multi_job_control_profiling_metrics.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:partners",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/common:multi_vehicle_utils",
        "//fueling/profiling/common:sanity_check",
        "//fueling/profiling/control/feature_extraction:multi_job_control_feature_extraction_utils",
        "//fueling/profiling/control/grading_evaluation:multi_job_control_performance_grading_utils",
    ],
)
