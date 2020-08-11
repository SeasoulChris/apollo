load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
	name = "emergency_vehicle_data_crawler",
	srcs = ["emergency_vehicle_data_crawler.py"]
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
