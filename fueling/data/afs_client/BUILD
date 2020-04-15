package(default_visibility = ["//visibility:public"])

py_binary(
    name = "client",
    srcs = ["client.py"],
    deps = [
        "//apps/afs_data_service/proto:afs_data_service_pb2", 
        "//fueling/common:base_pipeline", 
        "//fueling/common:logging", 
        "//fueling/common/record/kinglong/cybertron/python:convert",
        "//fueling/common/record/kinglong/proto/modules:localization_pose_pb2",
        "//fueling/common:file_utils",
    ]
)
