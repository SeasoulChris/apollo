package(default_visibility = ["//visibility:public"])

filegroup(
    name = "resource",
    srcs = glob([
        "static/**",
        "templates/**",
    ]),
)

py_binary(
    name = "application",
    srcs = ["application.py"],
    data = [":resource"],
    deps = [
	"//apps/k8s/admin_console/common:filter",
        "//apps/k8s/admin_console/conf:develop_setting",
    ],
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    data = [":resource"],
    deps = [
        ":application",
        "//apps/k8s/admin_console/views:index",
        "//apps/k8s/admin_console/views:job",
        "//apps/k8s/admin_console/views:statistics",
    ],
)
