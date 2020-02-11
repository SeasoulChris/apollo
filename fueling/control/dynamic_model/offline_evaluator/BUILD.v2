package(default_visibility = ["//visibility:public"])

py_library(
    name = "holistic_model_evaluator",
    srcs = ["holistic_model_evaluator.py"],
    deps = [
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/data_generator:holistic_data_generator",
        "//fueling/control/proto:dynamic_model_evaluation_pb2",
    ],
)

py_library(
    name = "non_holistic_model_evaluator",
    srcs = ["non_holistic_model_evaluator.py"],
    deps = [
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/data_generator:non_holistic_data_generator",
        "//fueling/control/proto:dynamic_model_evaluation_pb2",
    ],
)
