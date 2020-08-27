load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "auto_label_pipeline",
    srcs = ["auto_label_pipeline.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "@yolov4",
    ],
)

py_binary(
    name = "record_image_parser",
    srcs = ["record_image_parser.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/data:record_parser",
        "//fueling/perception:decode_video",
        "@apollo//modules/canbus/proto:chassis_py_pb2",
        "@apollo//modules/localization/proto:localization_py_pb2",
    ],
)
