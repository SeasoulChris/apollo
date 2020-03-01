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

py_test(
    name = "generate_small_records_test",
    srcs = ["generate_small_records_test.py"],
    data = ["//fueling/demo/testdata:small_record"],
    deps = [
        ":generate_small_records",
    ]
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

py_test(
    name = "reorg_small_records_test",
    srcs = ["reorg_small_records_test.py"],
    deps = [
        ":reorg_small_records",
    ]
)
