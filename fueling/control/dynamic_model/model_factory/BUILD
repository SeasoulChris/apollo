load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "lstm_keras",
    srcs = ["lstm_keras.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/control/dynamic_model/conf:model_config",
    ],
)

py_library(
    name = "mlp_keras",
    srcs = ["mlp_keras.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/proto:fnn_model_py_pb2",
    ],
)
