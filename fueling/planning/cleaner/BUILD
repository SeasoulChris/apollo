filegroup(
    name = "task_list",
    srcs = ["task_list.txt"],
)

py_binary(
    name = "data_cleaner",
    srcs = ["data_cleaner.py"],
    data = [
        "//fueling/planning/cleaner:task_list",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "@planning_analytics",
    ],
)

# TODO(yifei): Fix it so that we can guarantee in future.
#py_test(
#    name = "data_cleaner_test",
#    srcs = ["data_cleaner_test.py"],
#    deps = [
#        ":data_cleaner",
#    ],
#)

py_binary(
    name = "empty_folder",
    srcs = [
        "empty_folder.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "delete_folder",
    srcs = [
        "delete_folder.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "publish_cleaned_data",
    srcs = [
        "publish_cleaned_data.py",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)
