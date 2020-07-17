package(default_visibility = ["//visibility:public"])

py_library(
    name = "anchor_generator_builder",
    srcs = ["anchor_generator_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:anchor_generator",
        "//fueling/perception/pointpillars/second/protos:box_coder_py_pb2"
    ],
)

py_library(
    name = "dataset_builder",
    srcs = ["dataset_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/data:dataset",
        "//fueling/perception/pointpillars/second/protos:input_reader_py_pb2",
        "//fueling/perception/pointpillars/second/data:preprocess",
        "//fueling/perception/pointpillars/second/core:box_np_ops",
        ":dbsampler_builder",
        "//fueling/perception/pointpillars/second/utils:config_tool",
        "//fueling/perception/pointpillars/second/data:kitti_dataset",
        "//fueling/perception/pointpillars/second/data:nuscenes_dataset",
    ],
)

py_library(
    name = "dbsampler_builder",
    srcs = ["dbsampler_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:sample_ops",
        "//fueling/perception/pointpillars/second/core:preprocess",
        ":preprocess_builder",
    ],
)

py_library(
    name = "preprocess_builder",
    srcs = ["preprocess_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:preprocess",
    ],
)

py_library(
    name = "similarity_calculator_builder",
    srcs = ["similarity_calculator_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:region_similarity",
        "//fueling/perception/pointpillars/second/protos:similarity_py_pb2",
    ],
)

py_library(
    name = "target_assigner_builder",
    srcs = ["target_assigner_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:target_assigner",
        "//fueling/perception/pointpillars/second/protos:target_py_pb2", 
        "//fueling/perception/pointpillars/second/protos:anchors_py_pb2",
        ":similarity_calculator_builder",
        ":anchor_generator_builder",
    ],
)

py_library(
    name = "voxel_builder",
    srcs = ["voxel_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:voxel_generator",
        "//fueling/perception/pointpillars/second/protos:voxel_generator_py_pb2", 
    ],
)
