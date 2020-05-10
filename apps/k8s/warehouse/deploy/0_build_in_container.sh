#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Build Apollo.
/apollo/apollo.sh build_py
cp -r /apollo/py_proto ./

# Build the app.
bazel build //apps/k8s/warehouse:index
cp -f /fuel/bazel-bin/apps/k8s/warehouse/index.zip ./
