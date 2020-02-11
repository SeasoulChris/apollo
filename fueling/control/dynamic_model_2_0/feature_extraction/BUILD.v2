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
