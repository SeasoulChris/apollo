package(default_visibility = ["//visibility:public"])

load("//tools:py_proto.bzl", "py_proto")

py_proto(
    name = "spark_submit_arg_pb2",
    src = "spark_submit_arg.proto"
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    deps = [
        ":spark_submit_arg_pb2",
        "//fueling/common:mongo_utils",
        "//fueling/common:proto_utils",
    ],
)

py_library(
    name = "client",
    srcs = ["client.py"],
    deps = [
        "//fueling/common:file_utils"
    ],
)
