py_binary(
    name = "data_cleaner",
    srcs = [
        "data_cleaner.py",
        "msg_freq_analyzer.py",
        "routing_update_analyzer.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_test(
    name = "data_cleaner_test",
    srcs = ["data_cleaner_test.py"],
    data = ["//fueling/demo/testdata:small_record"],
    deps = [
        ":data_cleaner",
    ]
)