package(default_visibility = ["//visibility:public"])

py_library(
    name = "evaluation_method_util",
    srcs = ["evaluation_method_util.py"],
    data = [
        "//fueling/profiling/conf:profiling_conf",
    ],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/profiling/common:stats_utils",
        "//fueling/profiling/conf:open_space_planner_conf",
        "//fueling/profiling/proto:open_space_planner_profiling_pb2",
    ],
)
