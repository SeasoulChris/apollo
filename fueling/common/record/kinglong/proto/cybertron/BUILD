package(default_visibility = ["//visibility:public"])
load("//tools:py_proto.bzl", "py_proto")

py_proto (
  name = "record_pb2",
  src = "record.proto",
  deps = [],
)

py_proto (
  name = "cyber_header_pb2",
  src = "cyber_header.proto",
)

py_proto (
  name = "dag_config_pb2",
  src = "dag_config.proto",
)

py_proto (
  name = "errcode_pb2",
  src = "errcode.proto",
  deps = [":cyber_header_pb2"],
)

py_proto (
  name = "meta_message_pb2",
  src = "meta_message.proto",
)

py_proto (
  name = "meta_stat_pb2",
  src = "meta_stat.proto",
)

py_proto (
  name = "metric_service_pb2",
  src = "metric_service.proto",
)

py_proto (
  name = "module_service_pb2",
  src = "module_service.proto",
)

py_proto (
  name = "parameter_pb2",
  src = "parameter.proto",
)

py_proto (
  name = "proto_desc_pb2",
  src = "proto_desc.proto",
)

py_proto (
  name = "rule_desc_pb2",
  src = "rule_desc.proto",
)

py_proto (
  name = "sim_time_pb2",
  src = "sim_time.proto",
)

py_proto (
  name = "simulator_pb2",
  src = "simulator.proto",
)

py_proto (
  name = "topology_changes_pb2",
  src = "topology_changes.proto",
)

py_proto (
  name = "whisper_pb2",
  src = "whisper.proto",
  deps = [":cyber_header_pb2"],
)

py_proto (
  name = "channel_filter_pb2",
  src = "channel_filter.proto",
  deps = [":cyber_header_pb2"],
)

py_proto (
  name = "channel_service_pb2",
  src = "channel_service.proto",
)

py_proto (
  name = "chatter_pb2",
  src = "chatter.proto",
  deps = [":cyber_header_pb2"],
)

py_proto (
  name = "common_config_pb2",
  src = "common_config.proto",
  deps = [":cyber_header_pb2"],
)

py_proto (
  name = "config_schema_pb2",
  src = "config_schema.proto",
)

py_proto (
  name = "header_pb2",
  src = "header.proto",
)

py_proto (
  name = "common_geometry_pb2",
  src = "common_geometry.proto",
  deps = [":header_pb2"],
)

py_proto (
  name = "sensor_imu_pb2",
  src = "sensor_imu.proto",
  deps = [
    ":common_geometry_pb2",
    ":header_pb2",
  ],
)
