package(default_visibility = ["//visibility:public"])

filegroup(
    name = "resource",
    srcs = glob([
        "static/**",
        "templates/**",
    ]),
)

py_library(
    name = "index",
    srcs = ["index.py"],
    data = [":resource"],
    deps = [
    ],
)

py_binary(
    name = "job",
    srcs = ["job.py"],
    data = [":resource"],
    deps = [
        "//apps/k8s/admin_console/controllers:job",
        "//apps/k8s/admin_console/db:job",
    ],
)
