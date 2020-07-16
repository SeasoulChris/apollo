package(default_visibility = ["//visibility:public"])

py_library(
    name = "feature_extraction",
    srcs = ["feature_extraction.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model/conf:model_config",
    ],
)

py_library(
    name = "holistic_data_generator",
    srcs = ["holistic_data_generator.py"],
    deps = [
        ":feature_extraction",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
        "@apollo//modules/control/proto:control_conf_py_pb2",
    ],
)

py_library(
    name = "non_holistic_data_generator",
    srcs = ["non_holistic_data_generator.py"],
    deps = [
        ":feature_extraction",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/utils:echo_lincoln",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
        "@apollo//modules/control/proto:control_conf_py_pb2",
    ],
)

py_library(
    name = "training_data_generator",
    srcs = ["training_data_generator.py"],
    deps = [
        ":holistic_data_generator",
        ":non_holistic_data_generator",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model/conf:model_config",
    ],
)
