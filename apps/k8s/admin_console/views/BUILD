package(default_visibility = ["//visibility:public"])

py_library(
    name = "index",
    srcs = ["index.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
    ],
)

py_binary(
    name = "job",
    srcs = ["job.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console/controllers:job",
        "//apps/k8s/admin_console/db:job_db",
    ],
)
