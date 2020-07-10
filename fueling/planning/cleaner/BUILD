filegroup(
    name = "task_list",
    srcs = ["task_list.txt"],
)

py_binary(
    name = "data_cleaner",
    srcs = [
        "data_cleaner.py",
        "agent_analyzer.py",
        "analyzer_chassis.py",
        "analyzer_hmi.py",
        "analyzer_localization.py",
        "analyzer_perception.py",
        "analyzer_prediction.py",
        "analyzer_routing.py",
        "msg_freq_analyzer.py",
        "routing_update_analyzer.py",
    ],
    data = [
        "//fueling/planning/analytics:analytics",
        "//fueling/planning/cleaner:task_list",
    ],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/planning/apollo_record_reader:apollo_record_reader",
        "//fueling/planning/map_reader:map_reader",
    ],
)

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
