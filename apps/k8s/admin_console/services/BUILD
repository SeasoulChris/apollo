load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "account_services",
    srcs = ["account_services.py"],
    data = [
        "//apps/k8s/admin_console:resource",
        "//apps/k8s/admin_console/conf:admin_json",
    ],
    deps = [
        "//apps/k8s/admin_console/controllers:account",
        "//apps/k8s/admin_console/utils:conf_utils",
        "//fueling/common:file_utils",
    ],
)

py_library(
    name = "job_services",
    srcs = ["job_services.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console/controllers:account",
        "//apps/k8s/admin_console/controllers:job",
        "//fueling/common:logging",
    ],
)
