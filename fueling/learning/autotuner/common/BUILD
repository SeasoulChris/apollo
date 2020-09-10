load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:dynamic_model_info_py_pb2",
    ],
)

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        ":utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/proto:dynamic_model_info_py_pb2",
        "//fueling/learning/autotuner/proto:tuner_param_config_py_pb2",
    ],
)
