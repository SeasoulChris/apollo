package(default_visibility = ["//visibility:public"])

py_binary(
    name = "calibration_multi_sensors",
    srcs = ["calibration_multi_sensors.py"],
    deps = [
        ":calibration_config",
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:partners",
    ]
)

py_library(
    name = "calibration_config",
    srcs = ["calibration_config.py"],
    deps = [
        "//fueling/common:file_utils",
    ]
)
