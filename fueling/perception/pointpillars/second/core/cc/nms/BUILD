load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "nms_kernel",
    srcs = ["nms_kernel.cu"],
    deps = [
        "@boost",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)
