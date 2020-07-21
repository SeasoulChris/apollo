load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "data_visualizer",
    srcs = [
        "data_visualizer.py",
        "mkz_plotter.py",
        "obstacle_plotter.py",
        "routing_plotter.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/planning/map_reader",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/localization/proto:localization_py_pb2",
        "@apollo//modules/perception/proto:perception_obstacle_py_pb2",
        "@apollo//modules/routing/proto:routing_py_pb2",
    ],
)
