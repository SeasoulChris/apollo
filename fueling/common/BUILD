load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_pipeline",
    srcs = ["base_pipeline.py"],
    deps = [
        ":context_utils",
        ":job_utils",
        ":kubectl_utils",
        ":mongo_utils",
        "//apps/k8s/spark_submitter:client",
        "//fueling/common/storage:bos_client",
        "//fueling/common/storage:filesystem",
    ],
)

py_library(
    name = "context_utils",
    srcs = ["context_utils.py"],
)

py_test(
    name = "context_utils_test",
    srcs = ["context_utils_test.py"],
    deps = [
        ":context_utils",
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
        ":proto_utils",
        "//fueling/data/proto:record_meta_py_pb2",
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
    name = "job_utils",
    srcs = ["job_utils.py"],
    deps = [
        ":file_utils",
        ":logging",
        ":mongo_utils",
    ],
)

py_library(
    name = "admin_utils",
    srcs = ["admin_utils.py"],
    deps = [
        ":logging",
        ":mongo_utils",
    ],
)

py_library(
    name = "account_utils",
    srcs = ["account_utils.py"],
    deps = [
        ":logging",
        ":mongo_utils",
    ],
)

py_library(
    name = "json_utils",
    srcs = ["json_utils.py"],
    deps = [
        ":file_utils",
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
        ":localization_utils",
        ":time_utils",
        "@apollo//cyber/proto:record_py_pb2",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/control/proto:control_cmd_py_pb2",
        "@apollo//modules/dreamview/proto:hmi_status_py_pb2",
        "@apollo//modules/drivers/proto:conti_radar_py_pb2",
        "@apollo//modules/drivers/proto:pointcloud_py_pb2",
        "@apollo//modules/drivers/proto:sensor_image_py_pb2",
        "@apollo//modules/localization/proto:localization_py_pb2",
        "@apollo//modules/planning/proto:planning_py_pb2",
        "@apollo//modules/prediction/proto:prediction_obstacle_py_pb2",
        "@apollo//modules/routing/proto:routing_py_pb2",
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
    ],
)

py_library(
    name = "spark_helper",
    srcs = ["spark_helper.py"],
    deps = [
        ":logging",
    ],
)

py_library(
    name = "spark_op",
    srcs = ["spark_op.py"],
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

py_library(
    name = "kubectl_utils",
    srcs = ["kubectl_utils.py"],
)

py_library(
    name = "localization_utils",
    srcs = ["localization_utils.py"],
)
