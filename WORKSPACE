workspace(name = "fuel")


# Import loaders.
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


# Import common rules with bazel_federation: https://github.com/bazelbuild/bazel-federation
http_archive(
    name = "bazel_federation",
    url = "file:///home/libs/bazel-federation-0.0.1.tar.gz",
    sha256 = "e9326b089c10b2a641099b1c366788f7df7c714ff71495a70f45b20c4fe1b521",
    strip_prefix = "bazel-federation-0.0.1",
)

# C++ rules.
load("@bazel_federation//:repositories.bzl", "rules_cc")
rules_cc()
load("@bazel_federation//setup:rules_cc.bzl", "rules_cc_setup")
rules_cc_setup()

# Python rules from latest code to have python3 supported.
git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python",
    # branch = "master",  # To update the repo, enable this and disable the commit ID.
    commit = "38f86fb55b698c51e8510c807489c9f4e047480e",
    shallow_since = "1575517988 -0500",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
load("@rules_python//python:pip.bzl", "pip_repositories")
pip_repositories()

# Install python deps.
load("@rules_python//python:pip.bzl", "pip3_import")
pip3_import(
   name = "default_deps",
   requirements = "//deps:default.txt",
)
load("@default_deps//:requirements.bzl", "pip_install")
pip_install()

# Proto rules: https://github.com/bazelbuild/rules_proto
http_archive(
    name = "rules_proto",
    # 2019-08-01
    url = "file:///home/libs/rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()
