package(default_visibility = ["//visibility:public"])

py_library(
    name = "args_utils",
    srcs = ["args_utils.py"],
    data = ["//apps/k8s/admin_console:resource"],
)

py_library(
    name = "conf_utils",
    srcs = ["conf_utils.py"],
    data = ["//apps/k8s/admin_console:resource"],
    deps = [
        "//apps/k8s/admin_console:application",
    ],
)

py_library(
    name = "pwd_utils",
    srcs = ["pwd_utils.py"],
    data = ["//apps/k8s/admin_console:resource"],
)

py_library(
    name = "time_utils",
    srcs = ["time_utils.py"],
    data = ["//apps/k8s/admin_console:resource"],
)
