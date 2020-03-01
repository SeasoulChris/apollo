load("@default_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "populate_utils",
    srcs = ["populate_utils.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common/storage:bos_client",
        #requirement("pyquaternion"),
    ],
)

py_library(
    name = "populate_frames",
    srcs = ["populate_frames.py"],
    deps = [
        ":populate_utils",
        "//fueling/common:base_pipeline",
        "//fueling/common:email_utils",
        "//fueling/common:record_utils",
    ],
)

#py_test(
#    name = "populate_frames_test",
#    srcs = ["populate_frames_test.py"],
#    deps = [
#        ":populate_frames",
#    ]
#)
