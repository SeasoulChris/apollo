package(default_visibility = ["//visibility:public"])

py_binary(
    name = "combine_data_for_tuning_and_labels",
    srcs = ["combine_data_for_tuning_and_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
    ]
)

py_binary(
    name = "combine_features_and_labels",
    srcs = ["combine_features_and_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
	"//fueling/common:logging",
    ]
)

py_binary(
    name = "copy_files",
    srcs = ["copy_files.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ]
)

py_binary(
    name = "data_for_learning",
    srcs = ["data_for_learning.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:mongo_utils",
	"//fueling/common:db_backed_utils",
	"//fueling/common:logging",
	"//fueling/common:record_utils",
    ]
)

py_binary(
    name = "data_for_tuning",
    srcs = ["data_for_tuning.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:mongo_utils",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
    ]
)

py_binary(
    name = "delete_dirs",
    srcs = ["delete_dirs.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ]
)

py_binary(
    name = "dump_feature_proto",
    srcs = ["dump_feature_proto.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:mongo_utils",
        "//fueling/common:db_backed_utils",
        "//fueling/common:record_utils",
    ]
)

py_binary(
    name = "evaluate_performance",
    srcs = ["evaluate_performance.py"],
    deps = [
        "//fueling/common:base_pipeline",
    ]
)

py_binary(
    name = "frame_env",
    srcs = ["frame_env.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:mongo_utils",
        "//fueling/common:db_backed_utils",
	"//fueling/common:logging",
        "//fueling/common:record_utils",
    ]
)

py_binary(
    name = "generate_labels",
    srcs = ["generate_labels.py"],
    deps = [
	"//fueling/common:base_pipeline",
	"//fueling/common:logging",
	"//fueling/prediction/common:online_to_offline",
    ]
)

py_binary(
    name = "merge_labels",
    srcs = ["merge_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
    ]
)

py_binary(
    name = "prediction_app_data_labeling",
    srcs = ["prediction_app_data_labeling.py"],
    deps = [
        "//fueling/common:base_pipeline",
	":dump_feature_proto",
	":generate_labels",
	":merge_labels",
    ]
)

py_binary(
    name = "prediction_app_junction_mlp_pipeline",
    srcs = ["prediction_app_junction_mlp_pipeline.py"],
    deps = [
        #TODO
    ]
)

py_binary(
    name = "prediction_app_performance_evaluation",
    srcs = ["prediction_app_performance_evaluation.py"],
    deps = [
        "//fueling/common:base_pipeline",
	":prediction_result",
	":evaluate_performance",
    ]
)

py_binary(
    name = "prediction_result",
    srcs = ["prediction_result.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:mongo_utils",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:record_utils",
    ]
)
