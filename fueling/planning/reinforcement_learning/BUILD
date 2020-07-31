load("@rules_python//python:defs.bzl", "py_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "rl_math_util",
    srcs = ["rl_math_util.py"],
)

py_library(
    name = "rl_algorithm",
    srcs = ["rl_algorithm.py"],
    deps = [
        ":rl_math_util",
        "//fueling/common:coord_utils",
        "//fueling/common:logging",
        "//fueling/learning:train_utils",
        "//fueling/planning/input_feature_preprocessor:chauffeur_net_feature_generator",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
        "@apollo//modules/planning/proto:planning_py_pb2",
    ],
)

py_library(
    name = "environment",
    srcs = ["environment.py"],
    deps = [
        ":rl_math_util",
        ":grading_result_py_pb2",
        "//fueling/common:coord_utils",
        "//fueling/common:logging",
        "//fueling/learning:train_utils",
        "//fueling/planning/input_feature_preprocessor:chauffeur_net_feature_generator",
        "@apollo//cyber/python/cyber_py3:cyber",
        "@apollo//cyber/python/cyber_py3:cyber_time",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
        "@apollo//modules/planning/proto:planning_py_pb2",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":rl_algorithm",
        ":environment",
        "//fueling/learning:train_utils",
    ],
)

proto_library(
    name = "grading_result_proto",
    srcs = ["grading_result.proto"],
    deps = [
        "@apollo//modules/common/proto:header_proto",
    ],
)

py_proto_library(
    name = "grading_result_py_pb2",
    deps = [
        ":grading_result_proto",
        "@apollo//modules/common/proto:header_py_pb2",
    ],
)
