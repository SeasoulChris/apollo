load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "resource",
    srcs = glob([
        "scripts/**",
        "ssl_keys/**",
        "templates/**",
    ]),
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    data = [":resource"],
    deps = [
        "//fueling/common:logging",
        "//fueling/common:partners",
    ],
)
