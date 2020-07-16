package(default_visibility = ["//visibility:public"])

py_library(
    name = "account",
    srcs = ["account.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
        "//apps/k8s/admin_console/common:paginator",
        "//apps/k8s/admin_console/controllers:account",
    ],
)

py_library(
    name = "admin",
    srcs = ["admin.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
        "//apps/k8s/admin_console/utils:pwd_utils",
        "//fueling/common:admin_utils",
    ],
)

py_library(
    name = "index",
    srcs = ["index.py"],
    data = ["//apps/k8s/admin_console:resource"],
)

py_binary(
    name = "job",
    srcs = ["job.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
        "//apps/k8s/admin_console/controllers:job",
        "//apps/k8s/admin_console/common:paginator",
        "//fueling/common:job_utils",
    ],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
        "//apps/k8s/admin_console/controllers:job",
        "//apps/k8s/admin_console/utils:time_utils",
        "//fueling/common:job_utils",
        "//fueling/common:logging",
    ],
)
