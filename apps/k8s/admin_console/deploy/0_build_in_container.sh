#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Build the app.
bazel build //apps/k8s/admin_console:index
cp -f /fuel/bazel-bin/apps/k8s/admin_console/index.zip ./
