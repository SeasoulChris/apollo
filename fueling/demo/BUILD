package(default_visibility = ["//visibility:public"])

py_binary(
    name = "simplest_demo",
    srcs = ["simplest_demo.py"],
    deps = [
        "//fueling/common:base_pipeline_v2",
    ]
)

py_test(
    name = "simplest_demo_test",
    srcs = ["simplest_demo_test.py"],
    deps = [
        ":simplest_demo",
    ]
)

py_binary(
    name = "count_msg_by_channel",
    srcs = ["count_msg_by_channel.py"],
    data = ["//fueling/demo/testdata:small_record"],
    deps = [
        "//fueling/common:base_pipeline_v2",
        "//fueling/common:file_utils",
    ]
)

py_binary(
    name = "gpu_training_with_pytorch",
    srcs = ["gpu_training_with_pytorch.py"],
    deps = [
        "//fueling/common:base_pipeline",
    ]
)

py_binary(
    name = "gpu_training_with_tensorflow",
    srcs = ["gpu_training_with_tensorflow.py"],
    deps = [
        "//fueling/common:base_pipeline",
    ]
)

py_binary(
    name = "stat_auto_mileage",
    srcs = ["stat_auto_mileage.py"],
    deps = [
        "//fueling/common:base_pipeline_v2",
        "//fueling/data:record_parser",
    ]
)

py_test(
    name = "stat_auto_mileage_test",
    srcs = ["stat_auto_mileage_test.py"],
    data = ["//fueling/demo/testdata:small_record"],
    deps = [
        ":stat_auto_mileage",
    ]
)
