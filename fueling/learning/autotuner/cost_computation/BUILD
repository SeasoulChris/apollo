package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_cost_computation",
    srcs = ["base_cost_computation.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/learning/autotuner/client:sim_client",
        "//fueling/learning/autotuner/proto:cost_computation_service_pb2",
   ],
)

py_binary(
    name = "cost_computation_server",
    srcs = ["cost_computation_server.py"],
    deps = [
        "//apps/k8s/spark_submitter:client",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/proto:cost_computation_service_pb2",
    ],
)

py_binary(
    name = "mrac_cost_computation",
    srcs = ["mrac_cost_computation.py"],
    deps = [
        ":base_cost_computation",
        "//apps/k8s/spark_submitter:client",
        "//fueling/common:proto_utils",
        "//fueling/profiling/control:multi_job_control_profiling_metrics",
    ],
)
