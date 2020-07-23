load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "renderer_utils",
    srcs = ["renderer_utils.py"],
)

py_library(
    name = "agent_box_img_renderer",
    srcs = ["agent_box_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "agent_poses_future_img_renderer",
    srcs = ["agent_poses_future_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "agent_poses_history_img_renderer",
    srcs = ["agent_poses_history_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "base_offroad_mask_img_renderer",
    srcs = ["base_offroad_mask_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/map/proto:map_road_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "base_roadmap_img_renderer",
    srcs = ["base_roadmap_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/map/proto:map_road_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "base_speedlimit_img_renderer",
    srcs = ["base_speedlimit_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "obstacle_history_img_renderer",
    srcs = ["obstacle_history_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "obstacle_predictions_img_renderer",
    srcs = ["obstacle_predictions_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "offroad_mask_img_renderer",
    srcs = ["offroad_mask_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "roadmap_img_renderer",
    srcs = ["roadmap_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "routing_img_renderer",
    srcs = ["routing_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "speed_limit_img_renderer",
    srcs = ["speed_limit_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "traffic_lights_img_renderer",
    srcs = ["traffic_lights_img_renderer.py"],
    deps = [
        ":renderer_utils",
        "//fueling/common:proto_utils",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_overlap_py_pb2",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/map/proto:map_signal_py_pb2",
        "@apollo//modules/perception/proto:traffic_light_detection_py_pb2",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
    ],
)

py_library(
    name = "chauffeur_net_feature_generator",
    srcs = ["chauffeur_net_feature_generator.py"],
    deps = [
        ":agent_box_img_renderer",
        ":agent_poses_future_img_renderer",
        ":agent_poses_history_img_renderer",
        ":base_offroad_mask_img_renderer",
        ":base_roadmap_img_renderer",
        ":base_speedlimit_img_renderer",
        ":obstacle_history_img_renderer",
        ":obstacle_predictions_img_renderer",
        ":offroad_mask_img_renderer",
        ":renderer_utils",
        ":roadmap_img_renderer",
        ":routing_img_renderer",
        ":speed_limit_img_renderer",
        ":traffic_lights_img_renderer",
        "@apollo//modules/map/proto:map_py_pb2",
        "@apollo//modules/map/proto:map_lane_py_pb2",
        "@apollo//modules/map/proto:map_road_py_pb2",
        "@apollo//modules/map/proto:map_overlap_py_pb2",
        "@apollo//modules/map/proto:map_signal_py_pb2",
        "@apollo//modules/planning/proto:learning_data_py_pb2",
        "@apollo//modules/planning/proto:planning_semantic_map_config_py_pb2",
        "@apollo//modules/perception/proto:traffic_light_detection_py_pb2",
    ],
)
