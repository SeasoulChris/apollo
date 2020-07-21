load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "semantic_map_single_image_dataset",
    srcs = ["semantic_map_single_image_dataset.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/learning:network_utils",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ],
)

py_library(
    name = "semantic_map_single_image_model",
    srcs = ["semantic_map_single_image_model.py"],
    deps = [
        "//fueling/common:coord_utils",
        "//fueling/learning:backbone",
        "//fueling/learning:network_utils",
        "//fueling/learning:train_utils",
    ],
)

py_binary(
    name = "semantic_map_single_image_pipeline",
    srcs = ["semantic_map_single_image_pipeline.py"],
    deps = [
        ":semantic_map_single_image_model",
        "//fueling/learning:train_utils",
    ],
)

py_binary(
    name = "convert_onnx",
    srcs = ["convert_onnx.py"],
    deps = [
        ":semantic_map_single_image_dataset",
        ":semantic_map_single_image_model",
    ],
)

py_test(
    name = "test_apollo_single_image_inference",
    srcs = ["test_apollo_single_image_inference.py"],
    data = glob(["test/**"]),
    deps = [
        ":semantic_map_single_image_dataset",
        ":semantic_map_single_image_model",
    ],
)
