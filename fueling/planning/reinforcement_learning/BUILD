load("@rules_python//python:defs.bzl", "py_library")

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

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":rl_algorithm",
        ":environment",
        "//fueling/learning:train_utils",
    ],
)
