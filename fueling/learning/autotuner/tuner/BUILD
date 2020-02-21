package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "bayesian_optimization_tuner",
    srcs = ["bayesian_optimization_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/learning/autotuner/client:cost_computation_client",
        "//fueling/learning/autotuner/proto:tuner_param_config_pb2",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        requirement("bayesian-optimization"),
    ],
)

py_binary(
    name = "optuna_tuner",
    srcs = ["optuna_tuner.py"],
    deps = [
        requirement("optuna"),
    ],
)
