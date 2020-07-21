load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "convert_bin_to_pb_txt",
    srcs = ["convert_bin_to_pb_txt.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)
