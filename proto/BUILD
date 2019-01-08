package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "record_data_proto",
    deps = [
        ":record_data_proto_lib",
    ],
)

proto_library(
    name = "record_data_proto_lib",
    srcs = ["record_data.proto"],
    deps = [
        "//cyber/proto:record_proto",
    ],
)

cc_proto_library(
    name = "record_io_proto",
    deps = [
        ":record_io_proto_lib",
    ],
)

proto_library(
    name = "record_io_proto_lib",
    srcs = ["record_io.proto"],
    deps = [
        ":record_data_proto_lib",
    ],
)
