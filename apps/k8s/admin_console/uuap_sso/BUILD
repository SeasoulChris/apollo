package(default_visibility = ["//visibility:public"])

py_binary(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":uuapsso",
    ],
)

py_binary(
    name = "uuapsso",
    srcs = ["uuapsso.py"],
    deps = [
        "//apps/k8s/admin_console:application",
    ],
)
