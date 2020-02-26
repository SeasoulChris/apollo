package(default_visibility = ["//visibility:public"])

py_library(
    name = "vehicle_dynamics_profiling_visualization",
    srcs = ["vehicle_dynamics_profiling_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/vehicle_dynamics/feature_visualization:vehicle_dynamics_feature_visualization_utils",
    ],
)

py_library(
    name = "vehicle_dynamics_profiling_metrics",
    srcs = ["vehicle_dynamics_profiling_metrics.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
        "//fueling/profiling/common:dir_utils",
        "//fueling/profiling/vehicle_dynamics/feature_extraction:vehicle_dynamics_feature_extraction_utils",
        "//fueling/profiling/vehicle_dynamics/grading_evaluation:vehicle_dynamics_grading_utils",
    ],
)
