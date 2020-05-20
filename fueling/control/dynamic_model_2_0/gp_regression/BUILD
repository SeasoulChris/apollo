package(default_visibility = ["//visibility:public"])

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    deps = [
        "//fueling/common:logging",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_binary(
    name = "dynamic_model_dataset",
    srcs = ["dynamic_model_dataset.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset",
        ":dynamic_model_dataset",
        ":gp_model",
        "//fueling/learning:train_utils",
    ],
)

py_library(
    name = "gp_model",
    srcs = ["gp_model.py"],
    deps = [
        ":encoder",
        "//fueling/common:logging",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_binary(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":dataset",
        ":encoder",
        ":gp_model",
    ],
)

py_binary(
    name = "file_split",
    srcs = ["file_split.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
    ],
)
