package(default_visibility = ["//visibility:public"])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "md5",
    hdrs = ["md5.h"],
)

cc_library(
    name = "define",
    hdrs = ["define.h"],
)

cc_library(
    name = "environment",
    hdrs = ["environment.h"],
    deps = [":macros"],
)

cc_library(
    name = "error_code",
    hdrs = ["error_code.h"],
)

cc_library(
    name = "file_util",
    srcs = ["file_util.cc"],
    hdrs = ["file_util.h"],
    deps = [
        ":define",
        ":error_code",
        ":macros",
        "//deps:boost",
    ],
)

cc_library(
    name = "data",
    srcs = ["data.cc"],
    hdrs = ["data.h"],
    deps = [
        ":define",
        ":error_code",
        ":macros",
    ],
)

cc_library(
    name = "protobuf_factory",
    srcs = ["protobuf_factory.cc"],
    hdrs = ["protobuf_factory.h"],
    deps = [
        ":define",
        ":error_code",
        ":macros",
        "//fueling/common/record/kinglong/proto/cybertron:proto_desc_proto",
    ],
)

cc_library(
    name = "message_base",
    srcs = ["message_base.cc"],
    hdrs = ["message_base.h"],
    deps = [
        ":define",
        ":error_code",
        ":macros",
        ":protobuf_factory",
        "//fueling/common/record/kinglong/proto/cybertron:cyber_header_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "raw_message",
    hdrs = ["raw_message.h"],
    deps = [
        ":message_base",
    ],
)

cc_library(
    name = "registerer",
    srcs = ["registerer.cc"],
    hdrs = ["registerer.h"],
    deps = [
        ":error_code",
        ":macros",
    ],
)

cc_library(
    name = "rw_lock_guard",
    hdrs = ["rw_lock_guard.h"],
)

cc_library(
    name = "time_conversion",
    hdrs = ["time_conversion.h"],
    deps = [
        ":macros",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        ":message_base",
    ],
)

cc_library(
    name = "gflags_manager",
    srcs = ["gflags_manager.cc"],
    hdrs = ["gflags_manager.h"],
    deps = [
        ":define",
        ":environment",
        ":error_code",
        ":file_util",
        ":macros",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_library(
    name = "conf_manager",
    srcs = ["conf_manager.cc"],
    hdrs = ["conf_manager.h"],
    deps = [
        ":define",
        ":environment",
        ":error_code",
        ":file_util",
        ":gflags_manager",
        ":macros",
        "@com_github_gflags_gflags//:gflags",
    ],
)
