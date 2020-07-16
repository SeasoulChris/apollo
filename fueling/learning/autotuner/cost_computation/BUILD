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
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/client:sim_client",
        "//fueling/learning/autotuner/common:utils",
        "//fueling/learning/autotuner/proto:cost_computation_service_py_pb2",
    ],
)

py_binary(
    name = "cost_computation_server",
    srcs = ["cost_computation_server.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/client:sim_client",
        "//fueling/learning/autotuner/cost_computation/job:k8s_cost_job",
        "//fueling/learning/autotuner/cost_computation/job:local_cost_job",
        "//fueling/learning/autotuner/proto:cost_computation_service_py_pb2",
    ],
)

py_binary(
    name = "profiling_cost_computation",
    srcs = ["profiling_cost_computation.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        ":base_cost_computation",
        "//apps/k8s/spark_submitter:client",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/proto:cost_computation_conf_py_pb2",
        "//fueling/learning/autotuner/proto:cost_computation_service_py_pb2",
        "//fueling/profiling/control:multi_job_control_profiling_metrics",
        "//fueling/profiling/open_space_planner:metrics",
    ],
)
