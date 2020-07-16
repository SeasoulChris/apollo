load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@apollo//tools:python_rules.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "module_service_cc_proto",
    deps = [
        ":module_service_proto",
    ],
)

proto_library(
    name = "module_service_proto",
    srcs = ["module_service.proto"],
)

py_proto_library(
    name = "module_service_py_pb2",
    deps = [
        ":module_service_proto",
    ],
)

cc_proto_library(
    name = "record_cc_proto",
    deps = [
        ":record_proto",
    ],
)

proto_library(
    name = "record_proto",
    srcs = ["record.proto"],
)

py_proto_library(
    name = "record_py_pb2",
    deps = [
        ":record_proto",
    ],
)

cc_proto_library(
    name = "metric_alarm_cc_proto",
    deps = [
        ":metric_alarm_proto",
    ],
)

proto_library(
    name = "metric_alarm_proto",
    srcs = ["metric_alarm.proto"],
)

py_proto_library(
    name = "metric_alarm_py_pb2",
    deps = [
        ":metric_alarm_proto",
    ],
)

cc_proto_library(
    name = "header_cc_proto",
    deps = [
        ":header_proto",
    ],
)

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

cc_proto_library(
    name = "sensor_imu_cc_proto",
    deps = [
        ":sensor_imu_proto",
    ],
)

proto_library(
    name = "sensor_imu_proto",
    srcs = ["sensor_imu.proto"],
    deps = [
        ":common_geometry_proto",
        ":header_proto",
    ],
)

py_proto_library(
    name = "sensor_imu_py_pb2",
    deps = [
        ":common_geometry_py_pb2",
        ":header_py_pb2",
        ":sensor_imu_proto",
    ],
)

cc_proto_library(
    name = "common_geometry_cc_proto",
    deps = [
        ":common_geometry_proto",
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

cc_proto_library(
    name = "rule_desc_cc_proto",
    deps = [
        ":rule_desc_proto",
    ],
)

proto_library(
    name = "rule_desc_proto",
    srcs = ["rule_desc.proto"],
)

py_proto_library(
    name = "rule_desc_py_pb2",
    deps = [
        ":rule_desc_proto",
    ],
)

cc_proto_library(
    name = "metric_service_cc_proto",
    deps = [
        ":metric_service_proto",
    ],
)

proto_library(
    name = "metric_service_proto",
    srcs = ["metric_service.proto"],
)

py_proto_library(
    name = "metric_service_py_pb2",
    deps = [
        ":metric_service_proto",
    ],
)

cc_proto_library(
    name = "topology_changes_cc_proto",
    deps = [
        ":topology_changes_proto",
    ],
)

proto_library(
    name = "topology_changes_proto",
    srcs = ["topology_changes.proto"],
)

py_proto_library(
    name = "topology_changes_py_pb2",
    deps = [
        ":topology_changes_proto",
    ],
)

cc_proto_library(
    name = "sim_time_cc_proto",
    deps = [
        ":sim_time_proto",
    ],
)

proto_library(
    name = "sim_time_proto",
    srcs = ["sim_time.proto"],
)

py_proto_library(
    name = "sim_time_py_pb2",
    deps = [
        ":sim_time_proto",
    ],
)

cc_proto_library(
    name = "simulator_cc_proto",
    deps = [
        ":simulator_proto",
    ],
)

proto_library(
    name = "simulator_proto",
    srcs = ["simulator.proto"],
)

py_proto_library(
    name = "simulator_py_pb2",
    deps = [
        ":simulator_proto",
    ],
)

cc_proto_library(
    name = "meta_stat_cc_proto",
    deps = [
        ":meta_stat_proto",
    ],
)

proto_library(
    name = "meta_stat_proto",
    srcs = ["meta_stat.proto"],
)

py_proto_library(
    name = "meta_stat_py_pb2",
    deps = [
        ":meta_stat_proto",
    ],
)

cc_proto_library(
    name = "cyber_header_cc_proto",
    deps = [
        ":cyber_header_proto",
    ],
)

proto_library(
    name = "cyber_header_proto",
    srcs = ["cyber_header.proto"],
)

py_proto_library(
    name = "cyber_header_py_pb2",
    deps = [
        ":cyber_header_proto",
    ],
)

cc_proto_library(
    name = "channel_filter_cc_proto",
    deps = [
        ":channel_filter_proto",
    ],
)

proto_library(
    name = "channel_filter_proto",
    srcs = ["channel_filter.proto"],
    deps = [
        ":cyber_header_proto",
    ],
)

py_proto_library(
    name = "channel_filter_py_pb2",
    deps = [
        ":channel_filter_proto",
        ":cyber_header_py_pb2",
    ],
)

cc_proto_library(
    name = "parameter_cc_proto",
    deps = [
        ":parameter_proto",
    ],
)

proto_library(
    name = "parameter_proto",
    srcs = ["parameter.proto"],
)

py_proto_library(
    name = "parameter_py_pb2",
    deps = [
        ":parameter_proto",
    ],
)

cc_proto_library(
    name = "dag_config_cc_proto",
    deps = [
        ":dag_config_proto",
    ],
)

proto_library(
    name = "dag_config_proto",
    srcs = ["dag_config.proto"],
)

py_proto_library(
    name = "dag_config_py_pb2",
    deps = [
        ":dag_config_proto",
    ],
)

cc_proto_library(
    name = "errcode_cc_proto",
    deps = [
        ":errcode_proto",
    ],
)

proto_library(
    name = "errcode_proto",
    srcs = ["errcode.proto"],
    deps = [
        ":cyber_header_proto",
    ],
)

py_proto_library(
    name = "errcode_py_pb2",
    deps = [
        ":cyber_header_py_pb2",
        ":errcode_proto",
    ],
)

cc_proto_library(
    name = "config_schema_cc_proto",
    deps = [
        ":config_schema_proto",
    ],
)

proto_library(
    name = "config_schema_proto",
    srcs = ["config_schema.proto"],
)

py_proto_library(
    name = "config_schema_py_pb2",
    deps = [
        ":config_schema_proto",
    ],
)

cc_proto_library(
    name = "meta_message_cc_proto",
    deps = [
        ":meta_message_proto",
    ],
)

proto_library(
    name = "meta_message_proto",
    srcs = ["meta_message.proto"],
)

py_proto_library(
    name = "meta_message_py_pb2",
    deps = [
        ":meta_message_proto",
    ],
)

cc_proto_library(
    name = "channel_service_cc_proto",
    deps = [
        ":channel_service_proto",
    ],
)

proto_library(
    name = "channel_service_proto",
    srcs = ["channel_service.proto"],
)

py_proto_library(
    name = "channel_service_py_pb2",
    deps = [
        ":channel_service_proto",
    ],
)

cc_proto_library(
    name = "proto_desc_cc_proto",
    deps = [
        ":proto_desc_proto",
    ],
)

proto_library(
    name = "proto_desc_proto",
    srcs = ["proto_desc.proto"],
)

py_proto_library(
    name = "proto_desc_py_pb2",
    deps = [
        ":proto_desc_proto",
    ],
)

cc_proto_library(
    name = "whisper_cc_proto",
    deps = [
        ":whisper_proto",
    ],
)

proto_library(
    name = "whisper_proto",
    srcs = ["whisper.proto"],
    deps = [
        ":cyber_header_proto",
    ],
)

py_proto_library(
    name = "whisper_py_pb2",
    deps = [
        ":cyber_header_py_pb2",
        ":whisper_proto",
    ],
)

cc_proto_library(
    name = "common_config_cc_proto",
    deps = [
        ":common_config_proto",
    ],
)

proto_library(
    name = "common_config_proto",
    srcs = ["common_config.proto"],
)

py_proto_library(
    name = "common_config_py_pb2",
    deps = [
        ":common_config_proto",
    ],
)

cc_proto_library(
    name = "chatter_cc_proto",
    deps = [
        ":chatter_proto",
    ],
)

proto_library(
    name = "chatter_proto",
    srcs = ["chatter.proto"],
    deps = [
        ":cyber_header_proto",
    ],
)

py_proto_library(
    name = "chatter_py_pb2",
    deps = [
        ":chatter_proto",
        ":cyber_header_py_pb2",
    ],
)
