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
