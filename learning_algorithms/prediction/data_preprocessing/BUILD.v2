package(default_visibility = ["//visibility:public"])

py_binary(
    name = "combine_features_and_labels",
    srcs = ["combine_features_and_labels.py"],
    deps = [
        ":features_labels_utils",
    ],
)

py_library(
    name = "features_labels_utils",
    srcs = ["features_labels_utils.py"],
)

py_binary(
    name = "merge_label_dicts",
    srcs = ["merge_label_dicts.py"],
    deps = [
        ":features_labels_utils",
    ],
)
