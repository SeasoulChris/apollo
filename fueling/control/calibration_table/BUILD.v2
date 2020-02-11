package(default_visibility = ["//visibility:public"])

py_binary(
    name = "multi_job_data_distribution",
    srcs = ["multi_job_data_distribution.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:partners",
        "//fueling/control/common:training_conf",
        "//fueling/control/common:multi_vehicle_plot_utils",
        "//fueling/control/common:multi_vehicle_utils",
    ],
)

py_binary(
    name = "multi_job_feature_extraction",
    srcs = ["multi_job_feature_extraction.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:file_utils",
        "//fueling/common:partners",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/common:time_utils",
        "//fueling/control/common:multi_job_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/common:sanity_check",
        "//fueling/control/common:training_conf",
        "//fueling/control/features:calibration_table_utils",
        "//fueling/control/features:feature_extraction_rdd_utils",
        "//fueling/control/features:feature_extraction_utils",
    ],
)

py_binary(
    name = "multi_job_result_visualization",
    srcs = ["multi_job_result_visualization.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:h5_utils",
        "//fueling/common:partners",
        "//fueling/control/common:multi_job_utils",
        "//fueling/control/common:multi_vehicle_plot_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/common:training_conf",
    ],
)

py_binary(
    name = "multi_job_train",
    srcs = ["multi_job_train.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:proto_utils",
        "//fueling/control/common:multi_job_utils",
        "//fueling/control/common:multi_vehicle_utils",
        "//fueling/control/common:training_conf",
        "//fueling/control/features:calibration_table_train_utils",
        "//fueling/control/features:calibration_table_utils",
        "//fueling/control/proto:calibration_table_pb2",
    ],
)

py_binary(
    name = "vehicle_calibration",
    srcs = ["vehicle_calibration.py"],
    deps = [
        ":multi_job_data_distribution",
        ":multi_job_feature_extraction",
        ":multi_job_result_visualization",
        ":multi_job_train",
    ],
)
