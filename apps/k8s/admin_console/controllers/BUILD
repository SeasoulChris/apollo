load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "account",
    srcs = ["account.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console/common:paginator",
        "//apps/k8s/admin_console/controllers:job",
        "//apps/k8s/admin_console/utils:time_utils",
        "//fueling/common:account_utils",
    ],
)

py_library(
    name = "job",
    srcs = ["job.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console/utils:time_utils",
        "//fueling/common:mongo_utils",
    ],
)
