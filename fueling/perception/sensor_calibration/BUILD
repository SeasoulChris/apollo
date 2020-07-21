load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "calibration_multi_sensors",
    srcs = ["calibration_multi_sensors.py"],
    deps = [
        ":calibration_config",
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:job_utils",
        "//fueling/common:partners",
        "//fueling/common:redis_utils",
    ],
)

py_library(
    name = "calibration_config",
    srcs = ["calibration_config.py"],
    deps = [
        "//fueling/common:file_utils",
    ],
)
