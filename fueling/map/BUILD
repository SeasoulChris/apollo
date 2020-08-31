load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        "//fueling/common:context_utils",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
        "@apollo//cyber/python/cyber_py3:record",
    ],
)

py_library(
    name = "generate_base_map",
    srcs = ["generate_base_map.py"],
    deps = [
        ":sanity_check",
        "//fueling/common:base_pipeline",
        "//fueling/common:context_utils",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
        "//fueling/common:spark_helper",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/map/proto:map_road_py_pb2",
    ],
)

py_test(
    name = "generate_base_map_test",
    srcs = ["generate_base_map_test.py"],
    deps = [
        ":generate_base_map",
    ],
)

py_library(
    name = "generate_local_map",
    srcs = ["generate_local_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:context_utils",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
        "//fueling/common:spark_helper",
    ],
)

py_test(
    name = "generate_local_map_test",
    srcs = ["generate_local_map_test.py"],
    deps = [
        ":generate_local_map",
    ],
)

py_library(
    name = "generate_sim_routing_map",
    srcs = ["generate_sim_routing_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:context_utils",
        "//fueling/common:job_utils",
    ],
)

py_test(
    name = "generate_sim_routing_map_test",
    srcs = ["generate_sim_routing_map_test.py"],
    deps = [
        ":generate_sim_routing_map",
    ],
)

py_binary(
    name = "generate_maps",
    srcs = ["generate_maps.py"],
    deps = [
        ":generate_base_map",
        ":generate_local_map",
        ":generate_sim_routing_map",
    ],
)
