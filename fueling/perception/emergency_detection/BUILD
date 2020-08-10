package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
	name = "emergency_vehicle_data_crawler",
	srcs = ["emergency_vehicle_data_crawler.py"]
)
py_binary(
    name = "emergency_detection_train_pipeline",
    srcs = ["emergency_detection_train_pipeline.py"],
    deps = [
        "@yolov4",
        "//fueling/common:base_pipeline"
    ]
)

py_binary(
    name = "emergency_detection_infer_pipeline",
    srcs = ["emergency_detection_infer_pipeline.py"],
    deps = [
        "@yolov4",
        "//fueling/common:base_pipeline"
    ]
)