package(default_visibility = ["//visibility:public"])

py_library(
    name = "multi_job_utils",
    srcs = ["multi_job_utils.py"],
    data = ["//fueling/control/conf"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:proto_utils",
        "//fueling/control/features:calibration_table_utils",
        "//fueling/control/proto:calibration_table_pb2",
    ],
)

py_library(
    name = "multi_vehicle_plot_utils",
    srcs = ["multi_vehicle_plot_utils.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/control/dynamic_model/conf:model_config",
    ],
)

py_library(
    name = "multi_vehicle_utils",
    srcs = ["multi_vehicle_utils.py"],
    data = ["//fueling/control/conf"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:proto_utils",
        "//fueling/control/proto:calibration_table_pb2",
    ],
)

py_library(
    name = "sanity_check",
    srcs = ["sanity_check.py"],
    deps = [
        ":multi_vehicle_utils",
        "//fueling/common:email_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
    ],
)

py_library(
    name = "training_conf",
    srcs = ["training_conf.py"],
)
