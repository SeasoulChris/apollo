package(default_visibility = ["//visibility:public"])

py_library(
    name = "box_coder_builder",
    srcs = ["box_coder_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/protos:box_coder_py_pb2",
        "//fueling/perception/pointpillars/second/pytorch/core:box_coders",
    ],
)

py_library(
    name = "input_reader_builder",
    srcs = ["input_reader_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/protos:input_reader_py_pb2",
        "//fueling/perception/pointpillars/second/builder:dataset_builder",
    ],
)

py_library(
    name = "lr_scheduler_builder",
    srcs = ["lr_scheduler_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/torchplus/train:learning_schedules_fastai",
    ],
)

py_library(
    name = "optimizer_builder",
    srcs = ["optimizer_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/torchplus/train:learning_schedules",
        "//fueling/perception/pointpillars/torchplus/train:optim",
        "//fueling/perception/pointpillars/torchplus/train:fastai_optim",
    ],
)

py_library(
    name = "losses_builder",
    srcs = ["losses_builder.py"],
    deps = [
        "//fueling/perception/pointpillars/second/protos:losses_py_pb2",
        "//fueling/perception/pointpillars/second/pytorch/core:losses",
        "//fueling/perception/pointpillars/second/pytorch/core:ghm_loss",
    ],
)

py_library(
    name = "second_builder",
    srcs = ["second_builder.py"],
    deps = [
        ":losses_builder",
        "//fueling/perception/pointpillars/second/protos:second_py_pb2",
        "//fueling/perception/pointpillars/second/pytorch/models:voxelnet",
	"//fueling/perception/pointpillars/second/pytorch/models:net_multi_head",
    ],
)
