load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cnn_fc_model",
    srcs = ["cnn_fc_model.py"],
)

py_library(
    name = "cnn_lstm_model",
    srcs = ["cnn_lstm_model.py"],
    deps = [
        "//fueling/learning:network_utils",
    ],
)

py_library(
    name = "conv_rnn_model",
    srcs = ["conv_rnn_model.py"],
)

py_library(
    name = "image_representation_loss",
    srcs = ["image_representation_loss.py"],
    deps = [
        "//fueling/common:logging",
    ],
)

py_library(
    name = "trajectory_point_displacement_loss",
    srcs = ["trajectory_point_displacement_loss.py"],
)

py_library(
    name = "model_export",
    srcs = ["model_export.py"],
    deps = [
        ":cnn_fc_model",
        ":cnn_lstm_model",
        ":conv_rnn_model",
        "//fueling/common:logging",
        "//fueling/learning:train_utils",
    ],
)
