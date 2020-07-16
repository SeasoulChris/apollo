load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "feature_key_proto",
    srcs = ["feature_key.proto"],
)

py_proto_library(
    name = "feature_key_py_pb2",
    deps = [
        ":feature_key_proto",
    ],
)

proto_library(
    name = "dynamic_model_evaluation_proto",
    srcs = ["dynamic_model_evaluation.proto"],
)

py_proto_library(
    name = "dynamic_model_evaluation_py_pb2",
    deps = [
        ":dynamic_model_evaluation_proto",
    ],
)

proto_library(
    name = "fnn_model_proto",
    srcs = ["fnn_model.proto"],
)

py_proto_library(
    name = "fnn_model_py_pb2",
    deps = [
        ":fnn_model_proto",
    ],
)

proto_library(
    name = "calibration_table_proto",
    srcs = ["calibration_table.proto"],
)

py_proto_library(
    name = "calibration_table_py_pb2",
    deps = [
        ":calibration_table_proto",
    ],
)
