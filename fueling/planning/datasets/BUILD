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
    name = "dump_learning_data",
    srcs = ["dump_learning_data.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_test(
    name = "dump_learning_data_test",
    srcs = ["dump_learning_data_test.py"],
    deps = [
        ":dump_learning_data",
    ],
)

py_binary(
    name = "generate_training_data",
    srcs = ["generate_training_data.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
    ],
)