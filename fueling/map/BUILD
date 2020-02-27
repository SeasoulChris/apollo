package(default_visibility = ["//visibility:public"])

py_binary(
    name = "generate_base_map",
    srcs = ["generate_base_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:record_utils",
        "//fueling/common:redis_utils",
    ]
)

py_binary(
    name = "generate_local_map",
    srcs = ["generate_local_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
        "//fueling/common:file_utils",
        "//fueling/common:email_utils",
        "//fueling/common:redis_utils",
        "//fueling/common:partners",
    ]
)

py_binary(
    name = "generate_sim_routing_map",
    srcs = ["generate_sim_routing_map.py"],
    deps = [
        "//fueling/common:base_pipeline",
    ]
)

py_binary(
    name = "generate_maps",
    srcs = ["generate_maps.py"],
    deps = [
        ":generate_base_map",
        ":generate_local_map",
        ":generate_sim_routing_map",
    ]
)
