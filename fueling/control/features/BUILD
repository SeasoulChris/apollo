load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "calibration_table_train_utils",
    srcs = ["calibration_table_train_utils.py"],
    deps = [
        ":calibration_table_utils",
        ":neural_network_tf",
        "@apollo//modules/control/proto:calibration_table_py_pb2",
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
        "//fueling/control/proto:calibration_table_py_pb2",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
    ],
)

py_library(
    name = "dir_utils",
    srcs = ["dir_utils.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:record_utils",
        "//fueling/common:spark_op",
    ],
)

py_library(
    name = "feature_extraction_rdd_utils",
    srcs = ["feature_extraction_rdd_utils.py"],
    deps = [
        ":feature_extraction_utils",
        "//fueling/common:spark_op",
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
        "//fueling/control/proto:feature_key_py_pb2",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/common/configs/proto:vehicle_config_py_pb2",
        "@apollo//modules/dreamview/proto:hmi_status_py_pb2",
        "@apollo//modules/localization/proto:localization_py_pb2",
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
