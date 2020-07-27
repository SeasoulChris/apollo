package(default_visibility = ["//visibility:public"])

py_library(
    name = "account_services",
    srcs = ["account_services.py"],
    data = ["//apps/k8s/admin_console:resource"],
)

py_library(
    name = "job_services",
    srcs = ["job_services.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console/controllers:job",
        "//fueling/common:logging",
    ],
)
