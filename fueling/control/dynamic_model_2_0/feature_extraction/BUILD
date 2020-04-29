package(default_visibility = ["//visibility:public"])

py_binary(
    name = "feature_extraction",
    srcs = ["feature_extraction.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/common:record_utils",
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
