package(default_visibility = ["//visibility:public"])

py_library(
    name = "pybind11_build",
    srcs = ["pybind11_build.py"],
    deps = [
        "//fueling/perception/pointpillars/second/utils:loader",
        "//fueling/perception/pointpillars/second/utils:find",
        ":command"
    ],
)

py_library(
    name = "command",
    srcs = ["command.py"],
    deps = [
        "//fueling/perception/pointpillars/second/utils:find",
    ],
)


