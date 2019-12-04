package(default_visibility = ["//visibility:public"])

proto_library(
    name = "helloworld_lib",
    srcs = [
        "helloworld.proto",
    ],
)


cc_proto_library(
    name = "helloworld_proto",
    deps = [
        ":helloworld_lib",
    ],
)
