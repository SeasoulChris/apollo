load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "upload_to_bos",
    srcs = ["upload_to_bos.cc"],
    deps = [
        "//cyber/common:file",
        "//external:gflags",
        "//modules/common/util:string_util",
    ],
)

cpplint()
