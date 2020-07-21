package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
    name = "optuna_tuner",
    srcs = ["optuna_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/learning/autotuner/client:cost_computation_client",
        "//fueling/learning/autotuner/proto:tuner_param_config_py_pb2",
        "@apollo//modules/control/proto:control_conf_py_pb2",
        "@apollo//modules/control/proto:gain_scheduler_conf_py_pb2",
        "@apollo//modules/control/proto:lat_controller_conf_py_pb2",
        "@apollo//modules/control/proto:leadlag_conf_py_pb2",
        "@apollo//modules/control/proto:lon_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mpc_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mrac_conf_py_pb2",
        "@apollo//modules/control/proto:pid_conf_py_pb2",
        requirement("psycopg2-binary"),
    ],
)

py_binary(
    name = "bayesian_optimization_visual_utils",
    srcs = ["bayesian_optimization_visual_utils.py"],
)

py_library(
    name = "base_tuner",
    srcs = ["base_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        "//fueling/common:file_utils",
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
        ":base_tuner",
        ":bayesian_optimization_visual_utils",
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:tuner_param_config_py_pb2",
        "@apollo//modules/control/proto:control_conf_py_pb2",
        "@apollo//modules/control/proto:gain_scheduler_conf_py_pb2",
        "@apollo//modules/control/proto:lat_controller_conf_py_pb2",
        "@apollo//modules/control/proto:leadlag_conf_py_pb2",
        "@apollo//modules/control/proto:lon_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mpc_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mrac_conf_py_pb2",
        "@apollo//modules/control/proto:pid_conf_py_pb2",
    ],
)

py_binary(
    name = "optuna_optimization_tuner",
    srcs = ["optuna_optimization_tuner.py"],
    data = [
        "//fueling/learning/autotuner/config",
    ],
    deps = [
        ":base_tuner",
        "//fueling/common:logging",
        "//fueling/learning/autotuner/proto:tuner_param_config_py_pb2",
        "@apollo//modules/control/proto:control_conf_py_pb2",
        "@apollo//modules/control/proto:gain_scheduler_conf_py_pb2",
        "@apollo//modules/control/proto:lat_controller_conf_py_pb2",
        "@apollo//modules/control/proto:leadlag_conf_py_pb2",
        "@apollo//modules/control/proto:lon_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mpc_controller_conf_py_pb2",
        "@apollo//modules/control/proto:mrac_conf_py_pb2",
        "@apollo//modules/control/proto:pid_conf_py_pb2",
        "@apollo//modules/planning/proto:planner_open_space_config_py_pb2",
    ],
)

py_binary(
    name = "optuna_super_tuner",
    srcs = ["optuna_super_tuner.py"],
    deps = [
        ":optuna_optimization_tuner",
        "//fueling/common:logging",
        requirement("psycopg2-binary"),
    ],
)
