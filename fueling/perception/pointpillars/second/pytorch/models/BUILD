load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "middle",
    srcs = ["middle.py"],
)

py_library(
    name = "net_multi_head",
    srcs = ["net_multi_head.py"],
    deps = [
        ":rpn",
        ":voxelnet",
    ],
)

py_library(
    name = "pointpillars",
    srcs = ["pointpillars.py"],
    deps = [
        ":middle",
        ":voxel_encoder",
        "//fueling/perception/pointpillars/torchplus:tools",
        "//fueling/perception/pointpillars/torchplus/nn/modules:common",
    ],
)

py_library(
    name = "rpn",
    srcs = ["rpn.py"],
    deps = [
        "//fueling/perception/pointpillars/torchplus/nn/modules:common",
        "//fueling/perception/pointpillars/torchplus/nn/modules:normalization",
    ],
)

py_library(
    name = "voxel_encoder",
    srcs = ["voxel_encoder.py"],
    deps = [
        "//fueling/perception/pointpillars/torchplus/nn/modules:common",
        "//fueling/perception/pointpillars/torchplus/nn/modules:normalization",
    ],
)

py_library(
    name = "voxelnet",
    srcs = ["voxelnet.py"],
    deps = [
        ":middle",
        ":pointpillars",
        ":rpn",
        ":voxel_encoder",
        "//fueling/perception/pointpillars/second/pytorch/core:box_torch_ops",
        "//fueling/perception/pointpillars/second/pytorch/core:losses",
        "//fueling/perception/pointpillars/torchplus:metrics",
        "//fueling/perception/pointpillars/torchplus/nn:functional",
    ],
)
