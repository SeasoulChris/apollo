package(default_visibility = ["//visibility:public"])

py_binary(
    name = "generate_small_records",
    srcs = ["generate_small_records.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "index_records",
    srcs = ["index_records.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:mongo_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
        "//fueling/data:record_parser",
    ],
)

py_binary(
    name = "reorg_small_records",
    srcs = ["reorg_small_records.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/common:time_utils",
    ],
)
