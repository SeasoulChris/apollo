load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "label_generation",
    srcs = ["label_generation.py"],
    data = ["//fueling/control/dynamic_model_2_0/label_generation/mlp_model"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:redis_utils",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
    ],
)

py_binary(
    name = "pipeline_label_generation",
    srcs = ["pipeline_label_generation.py"],
    deps = [
        ":label_generation",
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:spark_helper",
        "//fueling/common:spark_op",
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/feature_extraction:feature_extraction_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)
