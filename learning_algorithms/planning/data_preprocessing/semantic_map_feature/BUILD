package(default_visibility = ["//visibility:public"])

py_library(
    name = "speed_limit_img_renderer",
    srcs = ["speed_limit_img_renderer.py"],
)

py_library(
    name = "agent_poses_history_img_renderer",
    srcs = ["agent_poses_history_img_renderer.py"],
)

py_library(
    name = "base_roadmap_img_renderer",
    srcs = ["base_roadmap_img_renderer.py"],
)

py_library(
    name = "obstacles_img_renderer",
    srcs = ["obstacles_img_renderer.py"],
)

py_library(
    name = "routing_img_renderer",
    srcs = ["routing_img_renderer.py"],
)

py_library(
    name = "chauffeur_net_feature_generator",
    srcs = ["chauffeur_net_feature_generator.py"],
    deps = ["//learning_algorithms/planning/data_preprocessing/semantic_map_feature:basemap_img_renderer"],
)

py_library(
    name = "agent_box_img_renderer",
    srcs = ["agent_box_img_renderer.py"],
)

py_library(
    name = "traffic_lights_img_renderer",
    srcs = ["traffic_lights_img_renderer.py"],
)

py_library(
    name = "roadmap_img_renderer",
    srcs = ["roadmap_img_renderer.py"],
    deps = [
        "//fueling/common:proto_utils",
    ],
)
