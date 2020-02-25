package(default_visibility = ["//visibility:public"])

load("//tools:py_proto.bzl", "py_proto")

py_proto(
    name = "record_pb2",
    src = "record.proto",
)

cc_proto_library(
    name = "record_proto",
    deps = [
        ":record_proto_lib",
    ],
)

proto_library(
    name = "record_proto_lib",
    srcs = ["record.proto"],
)

py_proto(
    name = "cyber_header_pb2",
    src = "cyber_header.proto",
)

cc_proto_library(
    name = "cyber_header_proto",
    deps = [
        ":cyber_header_proto_lib",
    ],
)

proto_library(
    name = "cyber_header_proto_lib",
    srcs = ["cyber_header.proto"],
)

py_proto(
    name = "dag_config_pb2",
    src = "dag_config.proto",
)

cc_proto_library(
    name = "dag_config_proto",
    deps = [
        ":dag_config_proto_lib",
    ],
)

proto_library(
    name = "dag_config_proto_lib",
    srcs = ["dag_config.proto"],
)

py_proto(
    name = "errcode_pb2",
    src = "errcode.proto",
    deps = [":cyber_header_pb2"],
)

cc_proto_library(
    name = "errcode_proto",
    deps = [
        ":errcode_proto_lib",
    ],
)

proto_library(
    name = "errcode_proto_lib",
    srcs = ["errcode.proto"],
    deps = [
        ":cyber_header_proto_lib",
    ],
)

py_proto(
    name = "meta_message_pb2",
    src = "meta_message.proto",
)

cc_proto_library(
    name = "meta_message_proto",
    deps = [
        ":meta_message_proto_lib",
    ],
)

proto_library(
    name = "meta_message_proto_lib",
    srcs = ["meta_message.proto"],
)

py_proto(
    name = "meta_stat_pb2",
    src = "meta_stat.proto",
)

cc_proto_library(
    name = "meta_stat_proto",
    deps = [
        ":meta_stat_proto_lib",
    ],
)

proto_library(
    name = "meta_stat_proto_lib",
    srcs = ["meta_stat.proto"],
)

py_proto(
    name = "metric_service_pb2",
    src = "metric_service.proto",
)

cc_proto_library(
    name = "metric_service_proto",
    deps = [
        ":metric_service_proto_lib",
    ],
)

proto_library(
    name = "metric_service_proto_lib",
    srcs = ["metric_service.proto"],
)

py_proto(
    name = "module_service_pb2",
    src = "module_service.proto",
)

cc_proto_library(
    name = "module_service_proto",
    deps = [
        ":module_service_proto_lib",
    ],
)

proto_library(
    name = "module_service_proto_lib",
    srcs = ["module_service.proto"],
)

py_proto(
    name = "parameter_pb2",
    src = "parameter.proto",
)

cc_proto_library(
    name = "parameter_proto",
    deps = [
        ":parameter_proto_lib",
    ],
)

proto_library(
    name = "parameter_proto_lib",
    srcs = ["parameter.proto"],
)

py_proto(
    name = "proto_desc_pb2",
    src = "proto_desc.proto",
)

cc_proto_library(
    name = "proto_desc_proto",
    deps = [
        ":proto_desc_proto_lib",
    ],
)

proto_library(
    name = "proto_desc_proto_lib",
    srcs = ["proto_desc.proto"],
)

py_proto(
    name = "rule_desc_pb2",
    src = "rule_desc.proto",
)

cc_proto_library(
    name = "rule_desc_proto",
    deps = [
        ":rule_desc_proto_lib",
    ],
)

proto_library(
    name = "rule_desc_proto_lib",
    srcs = ["rule_desc.proto"],
)

py_proto(
    name = "sim_time_pb2",
    src = "sim_time.proto",
)

cc_proto_library(
    name = "sim_time_proto",
    deps = [
        ":sim_time_proto_lib",
    ],
)

proto_library(
    name = "sim_time_proto_lib",
    srcs = ["sim_time.proto"],
)

py_proto(
    name = "simulator_pb2",
    src = "simulator.proto",
)

cc_proto_library(
    name = "simulator_proto",
    deps = [
        ":simulator_proto_lib",
    ],
)

proto_library(
    name = "simulator_proto_lib",
    srcs = ["simulator.proto"],
)

py_proto(
    name = "topology_changes_pb2",
    src = "topology_changes.proto",
)

cc_proto_library(
    name = "topology_changes_proto",
    deps = [
        ":topology_changes_proto_lib",
    ],
)

proto_library(
    name = "topology_changes_proto_lib",
    srcs = ["topology_changes.proto"],
)

py_proto(
    name = "whisper_pb2",
    src = "whisper.proto",
    deps = [":cyber_header_pb2"],
)

cc_proto_library(
    name = "whisper_proto",
    deps = [
        ":whisper_proto_lib",
    ],
)

proto_library(
    name = "whisper_proto_lib",
    srcs = ["whisper.proto"],
    deps = [":cyber_header_proto_lib"],
)

py_proto(
    name = "channel_filter_pb2",
    src = "channel_filter.proto",
    deps = [":cyber_header_pb2"],
)

cc_proto_library(
    name = "channel_filter_proto",
    deps = [
        ":channel_filter_proto_lib",
    ],
)

proto_library(
    name = "channel_filter_proto_lib",
    srcs = ["channel_filter.proto"],
    deps = [":cyber_header_proto_lib"],
)

py_proto(
    name = "channel_service_pb2",
    src = "channel_service.proto",
)

cc_proto_library(
    name = "channel_service_proto",
    deps = [
        ":channel_service_proto_lib",
    ],
)

proto_library(
    name = "channel_service_proto_lib",
    srcs = ["channel_service.proto"],
)

py_proto(
    name = "chatter_pb2",
    src = "chatter.proto",
    deps = [":cyber_header_pb2"],
)

cc_proto_library(
    name = "chatter_proto",
    deps = [
        ":chatter_proto_lib",
    ],
)

proto_library(
    name = "chatter_proto_lib",
    srcs = ["chatter.proto"],
    deps = [":cyber_header_proto_lib"],
)

py_proto(
    name = "common_config_pb2",
    src = "common_config.proto",
    deps = [":cyber_header_pb2"],
)

cc_proto_library(
    name = "common_config_proto",
    deps = [
        ":common_config_proto_lib",
    ],
)

proto_library(
    name = "common_config_proto_lib",
    srcs = ["common_config.proto"],
)

py_proto(
    name = "config_schema_pb2",
    src = "config_schema.proto",
)

cc_proto_library(
    name = "config_schema_proto",
    deps = [
        ":config_schema_proto_lib",
    ],
)

proto_library(
    name = "config_schema_proto_lib",
    srcs = ["config_schema.proto"],
)

py_proto(
    name = "header_pb2",
    src = "header.proto",
)

cc_proto_library(
    name = "header_proto",
    deps = [
        ":header_proto_lib",
    ],
)

proto_library(
    name = "header_proto_lib",
    srcs = ["header.proto"],
)

py_proto(
    name = "common_geometry_pb2",
    src = "common_geometry.proto",
    deps = [":header_pb2"],
)

cc_proto_library(
    name = "common_geometry_proto",
    deps = [
        ":common_geometry_proto_lib",
    ],
)

proto_library(
    name = "common_geometry_proto_lib",
    srcs = ["common_geometry.proto"],
    deps = [":header_proto_lib"],
)

py_proto(
    name = "sensor_imu_pb2",
    src = "sensor_imu.proto",
    deps = [
        ":common_geometry_pb2",
        ":header_pb2",
    ],
)

cc_proto_library(
    name = "sensor_imu_proto",
    deps = [
        ":sensor_imu_proto_lib",
    ],
)

proto_library(
    name = "sensor_imu_proto_lib",
    srcs = ["sensor_imu.proto"],
    deps = [
        ":common_geometry_proto_lib",
        ":header_proto_lib",
    ],
)
