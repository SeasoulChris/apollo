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
http_archive(
    name = "rules_python",
    url = "file:///home/libs/rules_python-8e9004ee8360d541abfcbecb60ba8a6902a53047.tar.gz",
    sha256 = "701b8d84d05c8b867d510d1778bbe12cc6ac79d09274c6bd71db77f053c16bca",
    strip_prefix = "rules_python-8e9004ee8360d541abfcbecb60ba8a6902a53047",
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
   timeout = 1000,
   extra_pip_args = [
       # EXTRA_PIP_ARGS
   ],
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

http_archive(
    name = "com_github_gflags_gflags",
    url = "file:///home/libs/gflags-2.2.2.tar.gz",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
)

http_archive(
    name = "com_google_glog",
    url = "file:///home/libs/glog-0.4.0.tar.gz",
    sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
    strip_prefix = "glog-0.4.0",
)

# TODO(xiaoxq): We'll refer apollo as dependency soon.
# Import apollo.
#local_repository(
#    name = "apollo",
#    path = "/apollo",
#)
