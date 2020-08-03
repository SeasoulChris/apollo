load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "before_request",
    srcs = ["before_request.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
    ],
)
