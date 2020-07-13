package(default_visibility = ["//visibility:public"])

py_library(
    name = "audio_features_extraction",
    srcs = ["audio_features_extraction.py"],
    deps = [
        "//fueling/audio/py_audio_analysis:audio_feature_extraction",
        "//fueling/common:file_utils",
        "//fueling/learning:train_utils",
    ],
)

py_library(
    name = "audio_torch_models",
    srcs = ["audio_torch_models.py"],
    deps = [
        ":audio_features_extraction",
        "//fueling/common:file_utils",
        "//fueling/learning:train_utils",
    ],
)

py_library(
    name = "audio_svm_model",
    srcs = ["audio_svm_model.py"],
    deps = [
        "//fueling/audio/py_audio_analysis:audio_feature_extraction",
    ],
)

py_library(
    name = "audio_model_evaluation",
    srcs = ["audio_model_evaluation.py"],
    deps = [
        ":audio_torch_models",
        "//fueling/audio/py_audio_analysis:audio_feature_extraction",
        "//fueling/common:file_utils",
    ],
)

py_library(
    name = "audio_urbansound8k_data_organizer",
    srcs = ["audio_urbansound8k_data_organizer.py"],
    deps = [
        "//fueling/common:file_utils",
    ],
)

py_binary(
    name = "siren_net",
    srcs = ["siren_net.py"],
    deps = [
        "//fueling/common:file_utils",
    ],
)
