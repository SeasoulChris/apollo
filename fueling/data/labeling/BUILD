load("@default_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "populate_utils",
    srcs = ["populate_utils.py"],
    deps = [
        "//fueling/common:file_utils",
        "//fueling/common:logging",
        "//fueling/common/storage:bos_client",
        "//fueling/streaming:streaming_utils",
        "@apollo//cyber/python/cyber_py3:record",
        "@apollo//modules/data/proto:frame_py_pb2",
        "@apollo//modules/common/proto:geometry_py_pb2",
        "@apollo//modules/drivers/proto:conti_radar_py_pb2",
        "@apollo//modules/drivers/proto:pointcloud_py_pb2",
        "@apollo//modules/localization/proto:localization_py_pb2",
        # TODO(xiaoxq): Fix it.
        # requirement("pypcd"),
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

# TODO(xiaoxq): Fix it.
#py_test(
#    name = "populate_frames_test",
#    srcs = ["populate_frames_test.py"],
#    deps = [
#        ":populate_frames",
#    ]
#)
