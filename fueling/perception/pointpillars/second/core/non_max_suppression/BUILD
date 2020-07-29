load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "nms_cpu",
    srcs = ["nms_cpu.py"],
    deps = [
        "//fueling/perception/pointpillars/second/core:box_np_ops",
	"@perception_pointpillars",
    ],
)

py_library(
    name = "nms_gpu",
    srcs = ["nms_gpu.py"],
    deps = ["@perception_pointpillars"],
)
