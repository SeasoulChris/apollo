load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
)

py_library(
    name = "box_np_ops",
    srcs = ["box_np_ops.py"],
    deps = [
        ":geometry",
        "//fueling/perception/pointpillars/second/utils/buildtools:pybind11_build",
    ],
)

py_library(
    name = "box_coders",
    srcs = ["box_coders.py"],
    deps = [
        ":box_np_ops",
    ],
)

py_library(
    name = "inference",
    srcs = ["inference.py"],
    deps = [
        "//fueling/perception/pointpillars/second/data:kitti_common",
        "//fueling/perception/pointpillars/second/data:preprocess",
        "//fueling/perception/pointpillars/second/protos:pipeline_py_pb2",
    ],
)

py_library(
    name = "anchor_generator",
    srcs = ["anchor_generator.py"],
    deps = [
        ":box_np_ops",
    ],
)

py_library(
    name = "region_similarity",
    srcs = ["region_similarity.py"],
    deps = [
        ":box_np_ops",
    ],
)

py_library(
    name = "target_ops",
    srcs = ["target_ops.py"],
    deps = [
        ":box_np_ops",
    ],
)

py_library(
    name = "target_assigner",
    srcs = ["target_assigner.py"],
    deps = [
        ":box_np_ops",
        ":region_similarity",
        ":target_ops",
        "//fueling/perception/pointpillars/second/utils:timer",
    ],
)

py_library(
    name = "sample_ops",
    srcs = ["sample_ops.py"],
    deps = [
        ":box_np_ops",
        ":preprocess",
        "//fueling/perception/pointpillars/second/utils:check",
    ],
)

py_library(
    name = "voxel_generator",
    srcs = ["voxel_generator.py"],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        ":box_np_ops",
        ":geometry",
    ],
)
