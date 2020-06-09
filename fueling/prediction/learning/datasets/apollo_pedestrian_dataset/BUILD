package(default_visibility = ["//visibility:public"])

py_library(
    name = "data_for_learning_pb2",
    srcs = ["data_for_learning_pb2.py"],
)

py_library(
    name = "apollo_pedestrian_dataset",
    srcs = ["apollo_pedestrian_dataset.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/datasets/apollo_pedestrian_dataset:data_for_learning_pb2",
    ],
)
