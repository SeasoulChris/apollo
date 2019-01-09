load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "record_io_service",
    srcs = ["record_io_service.cc"],
    linkopts = [
        "-lgrpc++",
    ],
    deps = [
        "//cyber",
        "//external:gflags",
        "//modules/common/util:string_util",
        "//modules/data/fuel/proto:record_io_grpc",
    ],
)
