load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "test_cost_client",
    srcs = ["test_cost_client.py"],
    deps = [
        "//fueling/learning/autotuner/client:cost_computation_client",
        "//fueling/learning/autotuner/proto:dynamic_model_info_py_pb2",
    ],
)
