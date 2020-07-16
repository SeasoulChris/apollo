load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "helloworld_proto",
    srcs = ["helloworld.proto"],
)

py_proto_library(
    name = "helloworld_py_pb2",
    deps = [
        ":helloworld_proto",
    ],
)
