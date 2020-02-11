package(default_visibility = ["//visibility:public"])

py_binary(
    name = "dynamic_model_data_visualization",
    srcs = ["dynamic_model_data_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/control/common:multi_vehicle_plot_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/data_generator:feature_extraction",
    ],
)

py_binary(
    name = "dynamic_model_evaluation",
    srcs = ["dynamic_model_evaluation.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/offline_evaluator:holistic_model_evaluator",
        "//fueling/control/dynamic_model/offline_evaluator:non_holistic_model_evaluator",
    ],
)

py_binary(
    name = "dynamic_model_training",
    srcs = ["dynamic_model_training.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/dynamic_model:flag",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/data_generator:feature_extraction",
        "//fueling/control/dynamic_model/data_generator:training_data_generator",
        "//fueling/control/dynamic_model/model_factory:lstm_keras",
        "//fueling/control/dynamic_model/model_factory:mlp_keras",
    ],
)

py_binary(
    name = "dynamic_model",
    srcs = ["dynamic_model.py"],
    deps = [
        ":dynamic_model_training",
        ":dynamic_model_data_visualization",
        "//fueling/control/dynamic_model/feature_extraction:sample_set",
        "//fueling/control/dynamic_model/feature_extraction:uniform_set",
    ],
)

py_library(
    name = "flag",
    srcs = ["flag.py"],
)

py_binary(
    name = "IMU_evaluation",
    srcs = ["IMU_evaluation.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/dynamic_model/data_generator:feature_extraction",
        "//fueling/control/dynamic_model/offline_evaluator:non_holistic_model_evaluator",
    ],
)
