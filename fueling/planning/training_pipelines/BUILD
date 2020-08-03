load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "trajectory_imitation_model_training_pipeline",
    srcs = [
        "trajectory_imitation_model_training_pipeline.py",
    ],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/planning/datasets:img_in_traj_out_dataset",
        "//fueling/planning/input_feature_preprocessor:chauffeur_net_feature_generator",
        "//fueling/planning/models/trajectory_imitation:cnn_fc_model",
        "//fueling/planning/models/trajectory_imitation:cnn_lstm_model",
        "//fueling/planning/models/trajectory_imitation:conv_rnn_model",
        "//fueling/planning/models/trajectory_imitation:image_representation_loss",
        "//fueling/planning/models/trajectory_imitation:trajectory_point_displacement_loss",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_binary(
    name = "trajectory_imitation_model_evaluation_pipeline",
    srcs = [
        "trajectory_imitation_model_evaluation_pipeline.py",
    ],
    deps = [
        "//fueling/learning:train_utils",
        "//fueling/planning/datasets:img_in_traj_out_dataset",
        "//fueling/planning/input_feature_preprocessor:chauffeur_net_feature_generator",
        "//fueling/planning/models/trajectory_imitation:cnn_fc_model",
        "//fueling/planning/models/trajectory_imitation:cnn_lstm_model",
        "//fueling/planning/models/trajectory_imitation:conv_rnn_model",
        "//fueling/planning/models/trajectory_imitation:image_representation_loss",
        "//fueling/planning/models/trajectory_imitation:trajectory_point_displacement_loss",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_binary(
    name = "model_training_on_cloud",
    srcs = [
        "model_training_on_cloud.py",
    ],
    deps = [
        ":trajectory_imitation_model_training_pipeline",
        "//fueling/common:base_pipeline",
    ],
)
