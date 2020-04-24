package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")


py_binary(
    name = "optuna_tuner",
    srcs = ["optuna_tuner.py"],
    deps = [
      "//fueling/learning/autotuner/client:cost_computation_client",
      "//fueling/learning/autotuner/proto:tuner_param_config_pb2",
      "//fueling/common:file_utils",
      "//fueling/common:proto_utils",
    ]
)

py_binary(
    name = "bayesian_optimization_visual_utils",
    srcs = ["bayesian_optimization_visual_utils.py"],
    deps = [
        requirement("bayesian-optimization"),
    ],
)


py_binary(
    name = "base_tuner",
    srcs = ["base_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/client:cost_computation_client",
    ],
)


py_binary(
    name = "bayesian_optimization_tuner",
    srcs = ["bayesian_optimization_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:tuner_param_config_pb2",
        "//fueling/learning/autotuner/tuner:base_tuner",
        "//fueling/learning/autotuner/tuner:bayesian_optimization_visual_utils",
        requirement("bayesian-optimization"),
    ],
)


py_binary(
    name = "optuna_optimization_tuner",
    srcs = ["optuna_optimization_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:tuner_param_config_pb2",
        "//fueling/learning/autotuner/tuner:base_tuner",
    ],
)
