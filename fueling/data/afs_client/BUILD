package(default_visibility = ["//visibility:public"])

py_library(
    name = "client",
    srcs = ["client.py"],
    deps = [
        "//apps/afs_data_service/proto:afs_data_service_pb2", 
        "//fueling/common:logging", 
        "//fueling/data/afs_client:afs_record_writer",
    ]
)

py_library(
    name = "config",
    srcs = ["config.py"],
)

py_library(
    name = "conversions",
    srcs = ["conversions.py"],
    deps = [
        "//fueling/common:logging", 
        "//fueling/common/record/kinglong/cybertron/python:convert",
        "//fueling/common/record/kinglong/proto/modules:localization_pose_pb2",
        "//fueling/common/record/kinglong/proto/modules:perception_obstacle_pb2",
    ]
)

py_library(
    name = "afs_record_writer",
    srcs = ["afs_record_writer.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging", 
        "//fueling/data/afs_client:config",
        "//fueling/data/afs_client:conversions",
    ]
)

py_binary(
    name = "scan_all_topics",
    srcs = ["scan_all_topics.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/data/afs_client:client",
    ]
)

py_binary(
    name = "scan_all_area_names",
    srcs = ["scan_all_area_names.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/data/afs_client:client",
    ]
)

py_binary(
    name = "scan_all_region_names",
    srcs = ["scan_all_region_names.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/data/afs_client:client",
    ]
)

py_binary(
    name = "copy_to_bos",
    srcs = ["copy_to_bos.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging", 
        "//fueling/data/afs_client:client",
        "//fueling/data/afs_client:config",
    ]
)


