#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Build Apollo.
/apollo/apollo.sh build_py
cp -r /apollo/py_proto ./

# Build the app.
cp ../BUILD.v2 ../BUILD
bazel build //apps/k8s/warehouse:index
cp /fuel/bazel-bin/apps/k8s/warehouse/index.zip ./
