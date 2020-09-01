workspace(name = "fuel")

local_repository(
    name = "apollo",
    path = "/apollo",
)

load("@apollo//tools:workspace.bzl", "apollo_repositories")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools:common.bzl", "clean_dep")

apollo_repositories()

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    # https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz
    url = "file:///fuel/deps/libs/bazel_skylib-0.9.0.tar.gz",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "rules_proto",
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
    # https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz
    url = "file:///fuel/deps/libs/rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

http_archive(
    name = "rules_python",
    sha256 = "701b8d84d05c8b867d510d1778bbe12cc6ac79d09274c6bd71db77f053c16bca",
    strip_prefix = "rules_python-8e9004ee8360d541abfcbecb60ba8a6902a53047",
    url = "file:///fuel/deps/libs/rules_python-8e9004ee8360d541abfcbecb60ba8a6902a53047.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:pip.bzl", "pip3_import", "pip_repositories")

pip_repositories()

pip3_import(
    name = "default_deps",
    timeout = 1000,
    extra_pip_args = [
        # EXTRA_PIP_ARGS
        # "-i https://pypi.tuna.tsinghua.edu.cn/simple",
    ],
    requirements = "//deps:default.txt",
)

load("@default_deps//:requirements.bzl", "pip_install")

pip_install()

# grpc
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "419dba362eaf8f1d36849ceee17c3e2ff8ff12ac666b42d3ff02a164ebe090e9",
    strip_prefix = "grpc-1.30.0",
    # https://github.com/grpc/grpc/archive/v1.30.0.tar.gz
    url = "file:///fuel/deps/libs/v1.30.0.tar.gz",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

http_archive(
    name = "planning_analytics",
    build_file = clean_dep("//third_party:planning_analytics.BUILD"),
    sha256 = "dfb4823b881d4033377f5382a682572e0c386060c6049066648bc487b7f532a5",
    strip_prefix = "planning_analytics",
    url = "file:///fuel/deps/libs/planning_analytics.zip",
)

http_archive(
    name = "perception_pointpillars",
    build_file = clean_dep("//third_party:perception_pointpillars.BUILD"),
    sha256 = "cb18d576990a0fd708d97735a6b461062f8056b581a936778aaea65ef46a8fde",
    strip_prefix = "perception_pointpillars",
    url = "file:///fuel/deps/libs/perception_pointpillars.zip",
)

new_local_repository(
    name = "yolov4",
    build_file = clean_dep("//third_party:yolov4.BUILD"),
    path = "/fuel/deps/libs/pytorch-YOLOv4",
)
