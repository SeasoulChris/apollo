load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "pointpillars_training_pipeline",
    srcs = [
        "pointpillars_training_pipeline.py",
    ],
    data = [
        "//testdata/perception/pointpillars:pointpillars_pretrained_model",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:job_utils",
        "//fueling/perception/pointpillars/second/pytorch:train",
    ],
)

py_test(
    name = "pointpillars_training_pipeline_test",
    srcs = ["pointpillars_training_pipeline_test.py"],
    deps = [
        ":pointpillars_training_pipeline",
    ]
)

py_binary(
    name = "create_data_pipeline",
    srcs = [
        "create_data_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:job_utils",
        "//fueling/perception/pointpillars/second:create_data",
    ],
)

py_binary(
    name = "create_kitti_data_pipeline",
    srcs = [
        "create_kitti_data_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:job_utils",
        "//fueling/perception/pointpillars/second:create_data",
    ],
)

py_test(
    name = "create_kitti_data_pipeline_test",
    srcs = ["create_kitti_data_pipeline_test.py",],
    deps = [
        ":create_kitti_data_pipeline",
    ]
)

py_binary(
    name = "pointpillars_export_onnx_pipeline",
    srcs = [
        "pointpillars_export_onnx_pipeline.py",
    ],
    data = [
        "//testdata/perception/pointpillars:pointpillars_pretrained_model",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:job_utils",
        "//fueling/perception/pointpillars/second/pytorch:trans_onnx",
    ],
)

py_test(
    name = "pointpillars_export_onnx_pipeline_test",
    srcs = ["pointpillars_export_onnx_pipeline_test.py"],
    deps = [
        ":pointpillars_export_onnx_pipeline",
    ]
)

py_binary(
    name = "pointpillars_end_to_end_pipeline",
    srcs = [
        "pointpillars_end_to_end_pipeline.py",
    ],
    deps = [
        ":create_kitti_data_pipeline",
        ":pointpillars_export_onnx_pipeline",
        ":pointpillars_training_pipeline",
    ],
)

py_binary(
    name = "pointpillars_eval_pipeline",
    srcs = [
        "pointpillars_eval_pipeline.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/perception/pointpillars/second/pytorch:train",
    ],
)
