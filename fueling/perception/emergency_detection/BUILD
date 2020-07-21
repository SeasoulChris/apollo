package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_binary(
	name = "emergency_vehicle_data_crawler",
	srcs = ["emergency_vehicle_data_crawler.py"]
)
py_binary(
    name = "emergency_detection_train_pipeline",
    srcs = ["emergency_detection_train_pipeline.py", 
            "dataset.py",
            "cfg.py",
            "models.py",
            "tool/torch_utils.py",
            "tool/config.py",
            "tool/utils.py",
            "tool/yolo_layer.py",
            "tool/darknet2pytorch.py",
            "tool/region_loss.py",
            "tool/tv_reference/utils.py",
            "tool/tv_reference/coco_utils.py",
            "tool/tv_reference/coco_eval.py",
            "tool/tv_reference/transforms.py"],
    deps = [
        "//fueling/common:base_pipeline"
    ]
)

py_binary(
    name = "emergency_detection_infer_pipeline",
    srcs = ["emergency_detection_infer_pipeline.py", 
            "tool/torch_utils.py",
            "tool/utils.py",
            "tool/yolo_layer.py"],
    deps = [
        "//fueling/common:base_pipeline"
    ]
)

py_binary(
    name = "dataloader_cloud_test",
    srcs = ["dataloader_cloud_test.py"],
    deps = [
        "//fueling/common:base_pipeline"
    ]
)

py_test(
    name = "emergency_detection_test",
    srcs = ["emergency_detection_test.py"],
    deps = [
        "//fueling/common:base_pipeline"
    ]
)
