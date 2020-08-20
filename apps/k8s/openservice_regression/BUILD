load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "openservice_regression",
    srcs = ["openservice_regression.py"],
    deps = [
        "//fueling/common:email_utils",
        "//fueling/common:logging",
    ],
)
