package(default_visibility = ["//visibility:public"])
load("//tools:grpc_library.bzl", "cc_grpc_library")

cc_proto_library(
    name = "record_data_proto",
    deps = [
        ":record_data_proto_lib",
    ],
)

proto_library(
    name = "record_data_proto_lib",
    srcs = ["record_data.proto"],
    deps = [
        "//cyber/proto:record_proto",
    ],
)

cc_grpc_library(
    name = "record_io_grpc",
    srcs = ["record_io.proto"],
    deps_cc = [
        ":record_data_proto",
    ],
    deps = [
        ":record_data_proto_lib",
    ],
)
