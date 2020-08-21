load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "feature_extraction",
    srcs = ["feature_extraction.py"],
    deps = [
        ":sample_set",
        ":uniform_set",
    ],
)

py_library(
    name = "sample_set",
    srcs = ["sample_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/common:spark_helper",
        "//fueling/common:spark_op",
        "//fueling/common:time_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/common:sanity_check",
        "//fueling/control/dynamic_model:flag",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:dir_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_library(
    name = "uniform_set",
    srcs = ["uniform_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/common:spark_helper",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/dynamic_model:flag",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:feature_extraction_utils",
    ],
)
