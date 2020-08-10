package(default_visibility = ["//visibility:public"])

load("@default_deps//:requirements.bzl", "requirement")

py_library(
    name = "YOLOv4",
    srcs = ["YOLOv4/train.py",
            "YOLOv4/dataset.py",
            "YOLOv4/cfg.py",
            "YOLOv4/models.py",
            "YOLOv4/tool/torch_utils.py",
            "YOLOv4/tool/config.py",
            "YOLOv4/tool/utils.py",
            "YOLOv4/tool/yolo_layer.py",
            "YOLOv4/tool/darknet2pytorch.py",
            "YOLOv4/tool/region_loss.py",
            "YOLOv4/tool/tv_reference/utils.py",
            "YOLOv4/tool/tv_reference/coco_utils.py",
            "YOLOv4/tool/tv_reference/coco_eval.py",
            "YOLOv4/tool/tv_reference/transforms.py"]
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
        ":YOLOv4",
        "//fueling/common:base_pipeline"
    ]
)