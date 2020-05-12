#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

bazel build //apps/k8s/spark_submitter:index
cp -f /fuel/bazel-bin/apps/k8s/spark_submitter/index.zip ./
