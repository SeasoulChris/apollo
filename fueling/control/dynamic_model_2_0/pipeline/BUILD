load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "data_splitter",
    srcs = ["data_splitter.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
    ],
)

py_binary(
    name = "training_pipeline",
    srcs = ["training_pipeline.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:distributed_data_parallel",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/gp_regression:dynamic_model_dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        "//fueling/control/dynamic_model_2_0/gp_regression:train",
        "//fueling/control/dynamic_model_2_0/gp_regression:train_utils",
    ],
)

py_binary(
    name = "training_test",
    srcs = ["training_test.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:distributed_data_parallel",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/gp_regression:dynamic_model_dataset",
        "//fueling/control/dynamic_model_2_0/gp_regression:encoder",
        "//fueling/control/dynamic_model_2_0/gp_regression:train",
        "//fueling/control/dynamic_model_2_0/gp_regression:train_utils",
    ],
)
