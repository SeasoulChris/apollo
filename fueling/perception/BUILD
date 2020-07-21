load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "benchmark_lidar",
    srcs = ["benchmark_lidar.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "decode_video",
    srcs = ["decode_video.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
        "//fueling/common:spark_helper",
        "//fueling/streaming:streaming_utils",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
