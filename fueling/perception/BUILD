package(default_visibility = ["//visibility:public"])

py_binary(
    name = "benchmark_lidar",
    srcs = ["benchmark_lidar.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ]
)

py_binary(
    name = "decode_video",
    srcs = ["decode_video.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
        "//fueling/streaming:streaming_utils",
    ]
)
