load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cost_computation_client",
    srcs = ["cost_computation_client.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning/autotuner/common:utils",
        "//fueling/learning/autotuner/proto:cost_computation_service_py_pb2",
        "//fueling/learning/autotuner/proto:cost_computation_service_py_pb2_grpc",
    ],
)

py_library(
    name = "sim_client",
    srcs = ["sim_client.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:git_info_py_pb2",
        "//fueling/learning/autotuner/proto:sim_service_py_pb2",
        "//fueling/learning/autotuner/proto:sim_service_py_pb2_grpc",
    ],
)
