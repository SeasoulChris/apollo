package(default_visibility = ["//visibility:public"])

filegroup(
    name = "resource",
    srcs = glob([
        "static/**",
        "templates/**",
    ]),
)

py_library(
    name = "display_util",
    srcs = ["display_util.py"],
    deps = [
        "//fueling/common:redis_utils",
    ],
)

py_binary(
    name = "index",
    srcs = ["index.py"],
    data = [":resource"],
    deps = [
        ":display_util",
        ":metrics_util",
        ":records_util",
        "//fueling/common:mongo_utils",
        "//fueling/common:proto_utils",
    ],
)

py_library(
    name = "metrics_util",
    srcs = ["metrics_util.py"],
    deps = [
        "//fueling/common:redis_utils",
    ],
)

py_binary(
    name = "plot_record_test",
    srcs = ["plot_record_test.py"],
    deps = [
        ":display_util",
        "//fueling/data:record_parser",
    ],
)

py_library(
    name = "records_util",
    srcs = ["records_util.py"],
    deps = [
        "//fueling/data/proto:record_meta_pb2",
    ],
)
