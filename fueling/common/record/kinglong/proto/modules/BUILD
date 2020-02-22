package(default_visibility = ["//visibility:public"])
load("//tools:py_proto.bzl", "py_proto")

py_proto (
  name = "header_pb2",
  src = "header.proto",
  deps = [],
)

py_proto (
  name = "common_geometry_pb2",
  src = "common_geometry.proto",
  deps = [":header_pb2"],
)

py_proto (
  name = "localization_raw_imu_pb2",
  src = "localization_raw_imu.proto",
  deps = [],
)

py_proto (
  name = "localization_status_pb2",
  src = "localization_status.proto",
  deps = [],
)

py_proto (
  name = "localization_pose_pb2",
  src = "localization_pose.proto",
  deps = [
    ":header_pb2",
    ":common_geometry_pb2",
    ":localization_raw_imu_pb2",
    ":localization_status_pb2",
  ],
)

py_proto (
  name = "perception_obstacle_pb2",
  src = "perception_obstacle.proto",
  deps = [":header_pb2"],
)
