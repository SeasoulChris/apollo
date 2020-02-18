package(default_visibility = ["//visibility:public"])

py_binary(
    name = "backward_sample_set",
    srcs = ["backward_sample_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/control/features:dir_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_library(
    name = "feature_extraction_visualization",
    srcs = ["feature_extraction_visualization.py"],
)

py_binary(
    name = "general_feature_extraction",
    srcs = ["general_feature_extraction.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/control/features:dir_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_binary(
    name = "golden_set",
    srcs = ["golden_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_binary(
    name = "sample_set_feature_extraction",
    srcs = ["sample_set_feature_extraction.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/control/features:dir_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_binary(
    name = "uniform_distribution_set",
    srcs = ["uniform_distribution_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:feature_extraction_utils",
    ],
)
