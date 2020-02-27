package(default_visibility = ["//visibility:public"])

py_library(
    name = "semantic_map_dataset",
    srcs = ["semantic_map_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//learning_algorithms/prediction/data_preprocessing/map_feature:online_mapping",
    ],
)

py_binary(
    name = "dump_feature_proto",
    srcs = ["dump_feature_proto.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)
