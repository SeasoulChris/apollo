package(default_visibility = ["//visibility:public"])

py_library(
    name = "bounding_rectangle",
    srcs = ["bounding_rectangle.py"],
    deps = [
        ":rotation2d",
        ":vector2d",
        ":util",
    ]
)

py_library(
    name = "configure",
    srcs = ["configure.py"],
)

py_library(
    name = "online_to_offline",
    srcs = ["online_to_offline.py"],
    deps = [
        ":bounding_rectangle",
        ":configure",
        "//fueling/common:proto_utils",
    ]
)

py_library(
    name = "rotation2d",
    srcs = ["rotation2d.py"],
    deps = [
        ":vector2d",
    ]
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_library(
    name = "vector2d",
    srcs = ["vector2d.py"],
)
