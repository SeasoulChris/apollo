package(default_visibility = ["//visibility:public"])

py_binary(
    name = "feature_extraction",
    srcs = ["feature_extraction.py"],
    deps = [
        ":sample_set",
        ":uniform_set",
    ],
)

py_binary(
    name = "sample_set",
    srcs = ["sample_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/common:time_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:feature_extraction_utils",
        "//fueling/control/features:dir_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
    ],
)

py_binary(
    name = "uniform_set",
    srcs = ["uniform_set.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:h5_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/dynamic_model/conf:model_config",
        "//fueling/control/features:feature_extraction_utils",
    ],
)
