package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "base_pipeline",
    srcs = ["base_pipeline.py"],
    deps = [
        "//apps/k8s/spark_submitter:client",
        "//fueling/common/storage:bos_client",
        "//fueling/common/storage:filesystem",
    ],
)

py_library(
    name = "base_pipeline_v2",
    srcs = ["base_pipeline_v2.py"],
    deps = [
        ":base_pipeline",
    ],
)

py_library(
    name = "coord_utils",
    srcs = ["coord_utils.py"],
)

py_library(
    name = "db_backed_utils",
    srcs = ["db_backed_utils.py"],
    deps = [
        "//fueling/data/proto:record_meta_pb2",
    ],
)

py_library(
    name = "email_utils",
    srcs = ["email_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "h5_utils",
    srcs = ["h5_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "json_utils",
    srcs = ["json_utils.py"],
    deps = [
        ":file_utils",
        ":proto_utils",
    ],
)

py_library(
    name = "logging",
    srcs = ["logging.py"],
)

py_library(
    name = "mongo_utils",
    srcs = ["mongo_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "partners",
    srcs = ["partners.py"],
)

py_library(
    name = "proto_utils",
    srcs = ["proto_utils.py"],
)

py_library(
    name = "record_utils",
    srcs = ["record_utils.py"],
    deps = [
        ":time_utils",
    ],
)

py_library(
    name = "redis_utils",
    srcs = ["redis_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "time_utils",
    srcs = ["time_utils.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "socket_utils",
    srcs = ["socket_utils.py"],
    deps = [
        ":logging",
        requirement("psutil"),
    ],
)

py_library(
    name = "distributed_data_parallel",
    srcs = ["distributed_data_parallel.py"],
    deps = [
        ":logging",
        ":redis_utils",
        ":socket_utils",
    ],
)
