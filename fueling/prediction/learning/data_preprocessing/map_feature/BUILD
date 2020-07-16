package(default_visibility = ["//visibility:public"])

py_library(
    name = "generate_img",
    srcs = ["generate_img.py"],
    deps = [
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
    ],
)

py_binary(
    name = "mapping",
    srcs = ["mapping.py"],
    deps = [
        "//fueling/prediction/common:configure",
        "@apollo//modules/map/proto:map_py_pb2",
    ],
)

py_binary(
    name = "obstacle_mapping",
    srcs = ["obstacle_mapping.py"],
    deps = [
        "//fueling/common:proto_utils",
        "@apollo//modules/prediction/proto:offline_features_py_pb2",
        #"@apollo//modules/prediction/proto:semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "online_mapping",
    srcs = ["online_mapping.py"],
    deps = [
        ":mapping",
        "//fueling/prediction/common:configure",
    ],
)

py_library(
    name = "semantic_map",
    srcs = ["semantic_map.py"],
    deps = [
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_py_pb2",
        #"@apollo//modules/prediction/proto:semantic_map_config_py_pb2",
    ],
)
