load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "combine_data_for_tuning_and_labels",
    srcs = ["combine_data_for_tuning_and_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:spark_op",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "combine_features_and_labels",
    srcs = ["combine_features_and_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:spark_op",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "copy_files",
    srcs = ["copy_files.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "data_for_learning",
    srcs = ["data_for_learning.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "data_for_tuning",
    srcs = ["data_for_tuning.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "delete_dirs",
    srcs = ["delete_dirs.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "dump_feature_proto",
    srcs = ["dump_feature_proto.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "evaluate_performance",
    srcs = ["evaluate_performance.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:spark_op",
        "@apollo//modules/prediction/proto:feature_py_pb2",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
        "@apollo//modules/prediction/proto:prediction_conf_py_pb2",
    ],
)

py_binary(
    name = "frame_env",
    srcs = ["frame_env.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "copy_map",
    srcs = ["copy_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
    ],
)

py_binary(
    name = "generate_labels",
    srcs = ["generate_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:spark_op",
        "//fueling/prediction/common:bounding_rectangle",
        "//fueling/prediction/common:online_to_offline",
    ],
)

py_binary(
    name = "merge_labels",
    srcs = ["merge_labels.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
    ],
)

py_binary(
    name = "prediction_app_data_labeling",
    srcs = ["prediction_app_data_labeling.py"],
    deps = [
        ":dump_feature_proto",
        ":generate_labels",
        ":merge_labels",
        "//fueling/common:base_pipeline",
    ],
)

py_binary(
    name = "prediction_app_junction_mlp_pipeline",
    srcs = ["prediction_app_junction_mlp_pipeline.py"],
    deps = [
        #TODO
    ],
)

py_binary(
    name = "prediction_app_performance_evaluation",
    srcs = ["prediction_app_performance_evaluation.py"],
    deps = [
        ":evaluate_performance",
        ":prediction_result",
        "//fueling/common:base_pipeline",
    ],
)

py_binary(
    name = "prediction_result",
    srcs = ["prediction_result.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:db_backed_utils",
        "//fueling/common:logging",
        "//fueling/common:mongo_utils",
        "//fueling/common:record_utils",
    ],
)

py_binary(
    name = "combine_frame_env_and_future_status",
    srcs = ["combine_frame_env_and_future_status.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/prediction/common:configure",
        "@apollo//modules/perception/proto:perception_obstacle_py_pb2",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "combine_frame_env_and_cross_road",
    srcs = ["combine_frame_env_and_cross_road.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common:proto_utils",
        "//fueling/common:record_utils",
        "//fueling/prediction/common:configure",
        "@apollo//modules/perception/proto:perception_obstacle_py_pb2",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "clean_and_balance_training_data",
    srcs = ["clean_and_balance_training_data.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:logging",
        "//fueling/common:spark_op",
        "//fueling/prediction/common:configure",
    ],
)

py_binary(
    name = "prediction_end_to_end",
    srcs = ["prediction_end_to_end.py"],
    deps = [
        ":combine_frame_env_and_future_status",
        ":copy_map",
        ":dump_feature_proto",
        ":frame_env",
        ":generate_labels",
        "//fueling/common:base_pipeline",
        "//fueling/common/storage:bos_client",
        "//fueling/prediction/learning/pipelines/pedestrian_trajectory_prediction:pedestrian_trajectory_training",
    ],
)
