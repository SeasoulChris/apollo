package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_cost_computation",
    srcs = ["base_cost_computation.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/learning/autotuner/client:sim_client",
        "//fueling/common:base_pipeline_v2",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "cost_computation_server",
    srcs = ["cost_computation_server.py"],
    deps = [
        "//apps/k8s/spark_submitter:client",
        "//fueling/learning/autotuner/proto:cost_computation_service_pb2",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
    ],
)

py_binary(
    name = "mrac_cost_computation",
    srcs = ["mrac_cost_computation.py"],
    deps = [
        ":base_cost_computation",
        "//fueling/learning/autotuner/proto:cost_computation_service_pb2",
        "//fueling/common:proto_utils",
    ],
)
