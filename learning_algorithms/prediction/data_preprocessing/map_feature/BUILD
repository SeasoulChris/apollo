package(default_visibility = ["//visibility:public"])

py_library(
    name = "generate_img",
    srcs = ["generate_img.py"],
)

py_binary(
    name = "mapping",
    srcs = ["mapping.py"],
)

py_binary(
    name = "obstacle_mapping",
    srcs = ["obstacle_mapping.py"],
    deps = [
        "//fueling/common:proto_utils",
    ],
)

py_library(
    name = "online_mapping",
    srcs = ["online_mapping.py"],
    deps = [
        ":mapping",
    ],
)

py_library(
    name = "semantic_map",
    srcs = ["semantic_map.py"],
    deps = [
        "//fueling/common:proto_utils",
    ],
)
