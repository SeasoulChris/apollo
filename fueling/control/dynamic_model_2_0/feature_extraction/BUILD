package(default_visibility = ["//visibility:public"])

py_library(
    name = "interpolation_message",
    srcs = ["interpolation_message.py"],
    deps = [
        "//fueling/control/dynamic_model_2_0/conf:model_conf",
        "//fueling/control/dynamic_model_2_0/feature_extraction:feature_extraction_utils",
    ],
)

py_library(
    name = "feature_extraction_utils",
    srcs = ["feature_extraction_utils.py"],
)

py_binary(
    name = "feature_extraction",
    srcs = ["feature_extraction.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/common:record_utils",
        "//fueling/control/dynamic_model_2_0/feature_extraction:feature_extraction_utils",
        "//fueling/control/dynamic_model_2_0/feature_extraction:interpolation_message",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_binary(
    name = "feature_extractor",
    srcs = ["feature_extractor.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "pipeline_feature_extraction",
    srcs = ["pipeline_feature_extraction.py"],
    deps = [
        ":feature_extractor",
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
    ],
)
