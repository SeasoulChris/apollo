package(default_visibility = ["//visibility:public"])

py_library(
    name = "open_space_profiling",
    srcs = ["open_space_profiling.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/profiling/open_space_planner:metrics",
    ],
)

py_binary(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
        "//fueling/profiling/control/feature_visualization:control_feature_visualization_utils",
        "//fueling/profiling/open_space_planner/feature_extraction:feature_extraction_utils",
        "//fueling/profiling/open_space_planner/metrics_utils:evaluation_method_util",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    data = [
        "//fueling/profiling/open_space_planner/testdata:records",
    ],
    deps = [
        ":metrics",
    ],
)
