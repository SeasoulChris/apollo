package(default_visibility = ["//visibility:public"])

py_library(
    name = "populate_utils",
    srcs = ["populate_utils.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common/storage:bos_client",
    ],
)

py_library(
    name = "populate_frames",
    srcs = ["populate_frames.py"],
    deps = [
        ":populate_utils",
    ],
)
