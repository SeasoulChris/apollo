package(default_visibility = ["//visibility:public"])
load("//tools:grpc_library.bzl", "cc_grpc_library")

cc_grpc_library(
    name = "record_io_grpc",
    srcs = ["record_io.proto"],
    deps_cc = [
        "//cyber/proto:record_cc_proto",
    ],
    deps = [
        "//cyber/proto:record_proto",
    ],
)
