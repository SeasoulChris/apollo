package(default_visibility = ["//visibility:public"])

py_binary(
    name = "dreamland",
    srcs = ["dreamland.py"],
    deps = [
        "//fueling/planning/stability/grading:planning_stability_grader",
    ]
)

py_library(
    name = "imu_speed_acc",
    srcs = ["imu_speed_acc.py"],
    deps = [
        ":imu_speed",
    ],
)

py_library(
    name = "imu_speed_jerk",
    srcs = ["imu_speed_jerk.py"],
    deps = [
        ":imu_speed_acc",
    ],
)

py_library(
    name = "imu_speed",
    srcs = ["imu_speed.py"],
)

py_library(
    name = "speed_jerk_stability",
    srcs = ["speed_jerk_stability.py"],
    deps = [
        ":imu_speed_jerk",
    ],
)
