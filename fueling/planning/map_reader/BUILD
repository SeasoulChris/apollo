load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "map_reader",
    srcs = ["map_reader.py"],
    deps = [
        "//fueling/common:redis_utils",
    ],
)

py_binary(
    name = "index_map_to_redis",
    srcs = [
        "index_map_to_redis.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:redis_utils",
        "@apollo//modules/map/proto:map_py_pb2",
    ],
)
