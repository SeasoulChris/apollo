load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "semantic_map_dataset",
    srcs = ["semantic_map_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ],
)

py_binary(
    name = "learning_data_generator",
    srcs = ["learning_data_generator.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_test(
    name = "learning_data_generator_test",
    srcs = ["learning_data_generator_test.py"],
    deps = [
        ":learning_data_generator",
    ],
)

py_binary(
    name = "trajectory_evaluator",
    srcs = ["trajectory_evaluator.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
    ],
)

py_binary(
    name = "generate_training_data",
    srcs = ["generate_training_data.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)

py_binary(
    name = "label_generator",
    srcs = ["label_generator.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/planning/data_visualizer",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
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
        "//fueling/common:spark_op",
    ],
)

py_binary(
    name = "post_processor",
    srcs = ["post_processor.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)

py_binary(
    name = "data_balancer",
    srcs = ["data_balancer.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)

py_library(
    name = "img_in_traj_out_dataset",
    srcs = ["img_in_traj_out_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/planning/datasets/semantic_map_feature:chauffeur_net_feature_generator",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_binary(
    name = "training_data_organizer",
    srcs = ["training_data_organizer.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)

py_binary(
    name = "trajectory_perturbation_synthesizer",
    srcs = ["trajectory_perturbation_synthesizer.py"],
)

py_binary(
    name = "trajectory_perturbation_synthesizer_pipeline",
    srcs = ["trajectory_perturbation_synthesizer_pipeline.py"],
    deps = [
        ":trajectory_perturbation_synthesizer",
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
    ],
)
