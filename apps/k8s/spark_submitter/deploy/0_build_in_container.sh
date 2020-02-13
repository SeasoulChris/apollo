#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

cp ../BUILD.v2 ../BUILD

bazel build //apps/k8s/spark_submitter:index
cp /fuel/bazel-bin/apps/k8s/spark_submitter/index.zip ./
