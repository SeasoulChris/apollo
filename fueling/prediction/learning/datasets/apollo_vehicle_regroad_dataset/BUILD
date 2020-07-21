load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "apollo_vehicle_regroad_dataset",
    srcs = ["apollo_vehicle_regroad_dataset.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/datasets/apollo_pedestrian_dataset:data_for_learning_pb2",
    ],
)

py_library(
    name = "test_apollo_vehicle_regroad_dataset",
    srcs = ["test_apollo_vehicle_regroad_dataset.py"],
)
