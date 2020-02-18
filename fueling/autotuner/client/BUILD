package(default_visibility = ["//visibility:public"])

py_library(
    name = "cost_computation_client",
    srcs = ["cost_computation_client.py"],
    deps = [
        "//fueling/autotuner/proto:cost_computation_service_pb2",
        "//fueling/common:logging",
    ],
)

py_library(
    name = "sim_client",
    srcs = ["sim_client.py"],
    deps = [
        "//fueling/autotuner/proto:git_info_pb2",
        "//fueling/autotuner/proto:sim_service_pb2",
        "//fueling/common:logging",
    ],
)
