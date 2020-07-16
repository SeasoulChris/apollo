load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "record_meta_proto",
    srcs = ["record_meta.proto"],
    deps = [
        "@apollo//cyber/proto:record_proto",
        "@apollo//modules/common/proto:drive_event_proto",
        "@apollo//modules/common/proto:geometry_proto",
        "@apollo//modules/dreamview/proto:hmi_status_proto",
    ],
)

py_proto_library(
    name = "record_meta_py_pb2",
    deps = [
        ":record_meta_proto",
        "@apollo//cyber/proto:record_py_pb2",
        "@apollo//modules/common/proto:drive_event_py_pb2",
        "@apollo//modules/common/proto:geometry_py_pb2",
        "@apollo//modules/dreamview/proto:hmi_status_py_pb2",
    ],
)
