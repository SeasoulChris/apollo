package(default_visibility = ["//visibility:public"])

py_binary(
    name = "daily_jobs",
    srcs = ["daily_jobs.py"],
    deps = [
        "//fueling/data/pipelines:generate_small_records",
        "//fueling/data/pipelines:index_records",
        "//fueling/data/pipelines:reorg_small_records",
        "//fueling/perception:decode_video",
    ]
)
