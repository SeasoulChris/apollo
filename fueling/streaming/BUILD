package(default_visibility = ["//visibility:public"])

py_library(
    name = "streaming_utils",
    srcs = ["streaming_utils.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
    ],
)

py_library(
    name = "serialize_utils",
    srcs = ["serialize_utils.py"],
    deps = [
        ":streaming_utils",
    ],
)

py_binary(
    name = "serialize_records",
    srcs = ["serialize_records.py"],
    deps = [
        ":serialize_utils",
        "//fueling/common:base_pipeline",
    ],
)
