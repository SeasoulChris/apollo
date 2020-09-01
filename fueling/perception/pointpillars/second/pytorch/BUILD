load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "//fueling/common:job_utils",
        "//fueling/perception/pointpillars/second/builder:target_assigner_builder",
        "//fueling/perception/pointpillars/second/builder:voxel_builder",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        "//fueling/perception/pointpillars/second/data:preprocess",
        "//fueling/perception/pointpillars/second/protos:pipeline_py_pb2",
        "//fueling/perception/pointpillars/second/pytorch/builder:box_coder_builder",
        "//fueling/perception/pointpillars/second/pytorch/builder:input_reader_builder",
        "//fueling/perception/pointpillars/second/pytorch/builder:lr_scheduler_builder",
        "//fueling/perception/pointpillars/second/pytorch/builder:optimizer_builder",
        "//fueling/perception/pointpillars/second/pytorch/builder:second_builder",
        "//fueling/perception/pointpillars/second/utils:log_tool",
        "//fueling/perception/pointpillars/second/utils:progress_bar",
        "//fueling/perception/pointpillars/torchplus/train:checkpoint",
        "//fueling/perception/pointpillars/torchplus/train:common",
        "//fueling/perception/pointpillars/torchplus/train:optim",
    ],
)

py_library(
    name = "trans_onnx",
    srcs = ["trans_onnx.py"],
    deps = [
        ":train",
        "//fueling/perception/pointpillars/second/protos:pipeline_py_pb2",
    ],
)
