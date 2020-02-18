package(default_visibility = ["//visibility:public"])

filegroup(
    name = "reference_grade_table",
    srcs = ["reference_grade_table.json"],
)

py_library(
    name = "imu_stability_grader",
    srcs = ["imu_stability_grader.py"],
    data = [
        ":reference_grade_table",
    ],
    deps = [
        "//fueling/planning/stability/libs:grade_table_utils",
        "//fueling/planning/stability/libs:imu_angular_velocity",
        "//fueling/planning/stability/libs:imu_speed_jerk",
        "//fueling/planning/stability/libs:record_reader",
    ]
)

py_library(
    name = "imu_stability_realtime_grader",
    srcs = ["imu_stability_realtime_grader.py"],
    deps = [
        ":imu_stability_grader",
        "//fueling/planning/stability/libs:imu_angular_velocity",
        "//fueling/planning/stability/libs:imu_speed_jerk",
    ]
)

py_library(
    name = "planning_stability_grader",
    srcs = ["planning_stability_grader.py"],
    data = [
        ":reference_grade_table",
    ],
    deps = [
        "//fueling/planning/stability/libs:grade_table_utils",
        "//fueling/planning/stability/libs:planning_av",
        "//fueling/planning/stability/libs:planning_jerk",
        "//fueling/planning/stability/libs:planning_lat_jerk",
        "//fueling/planning/stability/libs:record_reader",
    ]
)

py_library(
    name = "reference_grade_table_generator",
    srcs = ["reference_grade_table_generator.py"],
    data = [
        ":reference_grade_table",
    ],
    deps = [
        "//fueling/planning/stability/libs:grade_table_utils",
        "//fueling/planning/stability/libs:imu_angular_velocity",
        "//fueling/planning/stability/libs:imu_speed_jerk",
        "//fueling/planning/stability/libs:record_reader",
    ]
)
