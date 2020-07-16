load("@rules_proto//proto:defs.bzl", "proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "header_proto",
    srcs = ["header.proto"],
)

py_proto_library(
    name = "header_py_pb2",
    deps = [
        ":header_proto",
    ],
)

proto_library(
    name = "common_geometry_proto",
    srcs = ["common_geometry.proto"],
    deps = [
        ":header_proto",
    ],
)

py_proto_library(
    name = "common_geometry_py_pb2",
    deps = [
        ":common_geometry_proto",
        ":header_py_pb2",
    ],
)

proto_library(
    name = "localization_status_proto",
    srcs = ["localization_status.proto"],
)

py_proto_library(
    name = "localization_status_py_pb2",
    deps = [
        ":localization_status_proto",
    ],
)

proto_library(
    name = "perception_obstacle_proto",
    srcs = ["perception_obstacle.proto"],
    deps = [
        ":header_proto",
    ],
)

py_proto_library(
    name = "perception_obstacle_py_pb2",
    deps = [
        ":header_py_pb2",
        ":perception_obstacle_proto",
    ],
)

proto_library(
    name = "localization_raw_imu_proto",
    srcs = ["localization_raw_imu.proto"],
)

py_proto_library(
    name = "localization_raw_imu_py_pb2",
    deps = [
        ":localization_raw_imu_proto",
    ],
)

proto_library(
    name = "localization_pose_proto",
    srcs = ["localization_pose.proto"],
    deps = [
        ":common_geometry_proto",
        ":header_proto",
        ":localization_raw_imu_proto",
        ":localization_status_proto",
    ],
)

py_proto_library(
    name = "localization_pose_py_pb2",
    deps = [
        ":common_geometry_py_pb2",
        ":header_py_pb2",
        ":localization_pose_proto",
        ":localization_raw_imu_py_pb2",
        ":localization_status_py_pb2",
    ],
)
