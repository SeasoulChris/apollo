load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "yolov4",
    data = glob(["cfg/yolov4*.cfg"]),
    srcs = glob(["**/*.py"]),
)
