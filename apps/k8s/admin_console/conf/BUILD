package(default_visibility = ["//visibility:public"])

filegroup(
    name = "admin_json",
    srcs = ["admin.json"],
)

py_library(
    name = "base_setting",
    srcs = ["base_setting.py"],
)

py_binary(
    name = "develop_setting",
    srcs = ["develop_setting.py"],
    deps = [ 
        ":base_setting",      
    ],
)

py_binary(
    name = "production_setting",
    srcs = ["production_setting.py"],
    deps = [ 
        ":base_setting",      
    ],
)

py_binary(
    name = "test_setting",
    srcs = ["test_setting.py"],
    deps = [ 
        ":base_setting",      
    ],
)
