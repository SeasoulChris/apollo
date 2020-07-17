package(default_visibility = ["//visibility:public"])

py_library(
    name = "box_coders",
    srcs = ["box_coders.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_coders",
        ":box_torch_ops",
    ],
)

py_library(
    name = "box_torch_ops",
    srcs = ["box_torch_ops.py"],
    deps = [
        "//fueling/perception/pointpillars/torchplus:tools",
        "//fueling/perception/pointpillars/second/core/non_max_suppression:nms_gpu",
        "//fueling/perception/pointpillars/second/core/non_max_suppression:nms_cpu",
    ],
)

py_library(
    name = "ghm_loss",
    srcs = ["ghm_loss.py"],
    deps = [
        ":losses",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
)





