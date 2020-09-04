load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "calibration_multi_sensors",
    srcs = ["calibration_multi_sensors.py"],
    deps = [
        ":calibration_config",
        ":sanity_check",
        "//fueling/common:base_pipeline",
        "//fueling/common:context_utils",
        "//fueling/common:email_utils",
        "//fueling/common:job_utils",
        "//fueling/common:redis_utils",
    ],
)

py_test(
    name = "calibration_multi_sensors_test",
    srcs = ["calibration_multi_sensors_test.py"],
    deps = [
        ":calibration_multi_sensors",
    ],
)

py_library(
    name = "calibration_config",
    srcs = ["calibration_config.py"],
    deps = [
        "//fueling/common:file_utils",
    ],
)

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        "//fueling/common:context_utils",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
    ],
)
