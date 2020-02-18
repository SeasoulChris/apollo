package(default_visibility = ["//visibility:public"])

py_library(
    name = "calibration_table_train_utils",
    srcs = ["calibration_table_train_utils.py"],
    deps = [
        ":calibration_table_utils",
        ":neural_network_tf",
    ],
)

py_library(
    name = "calibration_table_utils",
    srcs = ["calibration_table_utils.py"],
    data = ["//fueling/control/conf"],
    deps = [
        ":filters",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:proto_utils",
        "//fueling/control/proto:calibration_table_pb2",
    ],
)

py_library(
    name = "dir_utils",
    srcs = ["dir_utils.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:record_utils",
    ],
)

py_library(
    name = "feature_extraction_rdd_utils",
    srcs = ["feature_extraction_rdd_utils.py"],
    deps = [
        ":feature_extraction_utils",
    ],
)

py_library(
    name = "feature_extraction_utils",
    srcs = ["feature_extraction_utils.py"],
    data = ["//fueling/control/conf"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/common:time_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/proto:feature_key_pb2",
    ],
)

py_library(
    name = "filters",
    srcs = ["filters.py"],
)

py_library(
    name = "neural_network_tf",
    srcs = ["neural_network_tf.py"],
)

py_library(
    name = "parameters",
    srcs = ["parameters.py"],
)
