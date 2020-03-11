package(default_visibility = ["//visibility:public"])

py_binary(
    name = "data_visualizer",
    srcs = [
        "data_visualizer.py",
        "mkz_plotter.py",
        "obstacle_plotter.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)
