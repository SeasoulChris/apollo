package(default_visibility = ["//visibility:public"])

py_library(
    name = "segment_social_lstm_model",
    srcs = ["segment_social_lstm_model.py"],
    deps = [
        "//fueling/prediction/learning/models/social_lstm_model",
    ],
)

py_library(
    name = "pipeline_social_lstm",
    srcs = ["pipeline_social_lstm.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/learning:train_utils",
        "//fueling/prediction/learning/models/social_lstm_model",
    ],
)

py_library(
    name = "social_lstm_model",
    srcs = ["social_lstm_model.py"],
)
