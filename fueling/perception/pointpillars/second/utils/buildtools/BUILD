load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pybind11_build",
    srcs = ["pybind11_build.py"],
    deps = [
        ":command",
        "//fueling/perception/pointpillars/second/utils:find",
        "//fueling/perception/pointpillars/second/utils:loader",
    ],
)

py_library(
    name = "command",
    srcs = ["command.py"],
    deps = [
        "//fueling/perception/pointpillars/second/utils:find",
    ],
)
