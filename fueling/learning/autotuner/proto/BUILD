load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_grpc_library", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "sim_service_proto",
    srcs = ["sim_service.proto"],
    deps = [
        ":dynamic_model_info_proto",
        ":git_info_proto",
    ],
)

py_proto_library(
    name = "sim_service_py_pb2",
    deps = [
        ":dynamic_model_info_py_pb2",
        ":git_info_py_pb2",
        ":sim_service_proto",
    ],
)

py_grpc_library(
    name = "sim_service_py_pb2_grpc",
    srcs = [":sim_service_proto"],
    deps = [":sim_service_py_pb2"],
)

proto_library(
    name = "dynamic_model_info_proto",
    srcs = ["dynamic_model_info.proto"],
)

py_proto_library(
    name = "dynamic_model_info_py_pb2",
    deps = [
        ":dynamic_model_info_proto",
    ],
)

proto_library(
    name = "tuner_param_config_proto",
    srcs = ["tuner_param_config.proto"],
    deps = [
        ":dynamic_model_info_proto",
        ":git_info_proto",
    ],
)

py_proto_library(
    name = "tuner_param_config_py_pb2",
    deps = [
        ":dynamic_model_info_py_pb2",
        ":git_info_py_pb2",
        ":tuner_param_config_proto",
    ],
)

proto_library(
    name = "cost_computation_service_proto",
    srcs = ["cost_computation_service.proto"],
    deps = [
        ":dynamic_model_info_proto",
        ":git_info_proto",
        ":sim_service_proto",
    ],
)

py_grpc_library(
    name = "cost_computation_service_py_pb2_grpc",
    srcs = [":cost_computation_service_proto"],
    deps = [":cost_computation_service_py_pb2"],
)

py_proto_library(
    name = "cost_computation_service_py_pb2",
    deps = [
        ":cost_computation_service_proto",
        ":dynamic_model_info_py_pb2",
        ":git_info_py_pb2",
        ":sim_service_py_pb2",
    ],
)

proto_library(
    name = "tuner_results_data_proto",
    srcs = ["tuner_results_data.proto"],
    deps = [
        ":tuner_param_config_proto",
    ],
)

py_proto_library(
    name = "tuner_results_data_py_pb2",
    deps = [
        ":tuner_param_config_py_pb2",
        ":tuner_results_data_proto",
    ],
)

proto_library(
    name = "cost_computation_conf_proto",
    srcs = ["cost_computation_conf.proto"],
)

py_proto_library(
    name = "cost_computation_conf_py_pb2",
    deps = [
        ":cost_computation_conf_proto",
    ],
)

proto_library(
    name = "git_info_proto",
    srcs = ["git_info.proto"],
)

py_proto_library(
    name = "git_info_py_pb2",
    deps = [
        ":git_info_proto",
    ],
)
