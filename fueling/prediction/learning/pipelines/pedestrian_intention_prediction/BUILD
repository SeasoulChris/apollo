package(default_visibility = ["//visibility:public"])

py_binary(
    name = "pedestrian_intention_dataset",
    srcs = ["pedestrian_intention_dataset.py"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:coord_utils",
        "//fueling/common:file_utils",
        "//fueling/prediction/learning/data_preprocessing/map_feature:online_mapping",
    ]
)
