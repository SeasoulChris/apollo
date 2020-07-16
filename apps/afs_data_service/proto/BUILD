load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_grpc_library", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "afs_data_service_proto",
    srcs = ["afs_data_service.proto"],
)

py_grpc_library(
    name = "afs_data_service_py_pb2_grpc",
    srcs = [":afs_data_service_proto"],
    deps = [":afs_data_service_py_pb2"],
)

py_proto_library(
    name = "afs_data_service_py_pb2",
    deps = [
        ":afs_data_service_proto",
    ],
)
