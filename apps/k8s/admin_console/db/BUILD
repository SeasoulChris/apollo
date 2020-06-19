package(default_visibility = ["//visibility:public"])

py_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        "//apps/k8s/admin_console:application",
    ],
)

py_library(
    name = "job",
    srcs = ["job.py"],
    deps = [
        ":base",
    ],
)
