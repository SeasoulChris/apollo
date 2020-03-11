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

py_binary(
    name = "label_generator",
    srcs = ["label_generator.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/planning/data_visualizer:data_visualizer",
    ],
)

py_binary(
    name = "output_data_generator",
    srcs = ["output_data_generator.py"],
    deps = [
        ":label_generator",
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
    ],
)
