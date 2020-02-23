package(default_visibility = ["//visibility:public"])

cc_library (
  name = "macros",
  hdrs = ["macros.h"],
  deps = [],
)

cc_library (
  name = "md5",
  hdrs = ["md5.h"],
  deps = [],
)

cc_library (
  name = "define",
  hdrs = ["define.h"],
)

cc_library (
  name = "environment",
  hdrs = ["environment.h"],
  deps = [":macros"],
)

cc_library (
  name = "error_code",
  hdrs = ["error_code.h"],
)

cc_library (
  name = "file_util",
  hdrs = ["file_util.h"],
  srcs = ["file_util.cc"],
  deps = [
    ":macros",
    ":error_code",
    ":define",
    "//deps:boost",
  ],
)

cc_library (
  name = "data",
  hdrs = ["data.h"],
  srcs = ["data.cc"],
  deps = [
    ":macros",
    ":error_code",
    ":define",
  ],
)

cc_library (
  name = "protobuf_factory",
  hdrs = ["protobuf_factory.h"],
  srcs = ["protobuf_factory.cc"],
  deps = [
    ":error_code",
    ":define",
    ":macros",
    "//fueling/common/record/kinglong/proto/cybertron:proto_desc_proto",
  ],
)

cc_library (
  name = "message_base",
  hdrs = ["message_base.h"],
  srcs = ["message_base.cc"],
  deps = [
    ":define",
    ":error_code",
    ":macros",
    ":protobuf_factory",
    "//fueling/common/record/kinglong/proto/cybertron:cyber_header_proto",
    "@com_google_protobuf//:protobuf",
  ],
)

cc_library (
  name = "raw_message",
  hdrs = ["raw_message.h"],
  deps = [
    ":message_base",
  ],
)

cc_library (
  name = "registerer",
  hdrs = ["registerer.h"],
  srcs = ["registerer.cc"],
  deps = [
    ":error_code",
    ":macros",
  ],
)

cc_library (
  name = "rw_lock_guard",
  hdrs = ["rw_lock_guard.h"],
  deps = [],
)

cc_library (
  name = "time_conversion",
  hdrs = ["time_conversion.h"],
  deps = [
    ":macros",
  ],
)

cc_library (
  name = "types",
  hdrs = ["types.h"],
  deps = [
    ":message_base",
  ],
)

cc_library (
  name = "gflags_manager",
  hdrs = ["gflags_manager.h"],
  srcs = ["gflags_manager.cc"],
  deps = [
    ":define",
    ":macros",
    ":file_util",
    ":environment",
    ":error_code",
    "@com_github_gflags_gflags//:gflags",
  ],
)

cc_library (
  name = "conf_manager",
  hdrs = ["conf_manager.h"],
  srcs = ["conf_manager.cc"],
  deps = [
    ":define",
    ":macros",
    ":file_util",
    ":environment",
    ":error_code",
    ":gflags_manager",
    "@com_github_gflags_gflags//:gflags",
  ],
)
