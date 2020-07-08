package(default_visibility = ["//visibility:public"])

py_library(
    name = "utilities",
    srcs = ["utilities.py"],
)

py_library(
    name = "audio_basic_io",
    srcs = ["audio_basic_io.py"],
)

py_library(
    name = "audio_feature_extraction",
    srcs = ["audio_feature_extraction.py"],
    deps = [
        ":audio_basic_io",
        ":utilities",
    ],
)
