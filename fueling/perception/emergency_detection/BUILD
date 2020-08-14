load("@rules_python//python:defs.bzl", "py_binary")
load("@default_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "emergency_vehicle_data_crawler",
    srcs = ["emergency_vehicle_data_crawler.py"],
    deps = [
        requirement("youtube-dl"),
        requirement("youtube-data-api"),
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "emergency_detection_train_pipeline",
    srcs = ["emergency_detection_train_pipeline.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "@yolov4",
    ],
)

py_binary(
    name = "emergency_detection_infer_pipeline",
    srcs = ["emergency_detection_infer_pipeline.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "@yolov4",
    ],
)
