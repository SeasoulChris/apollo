#!/usr/bin/env bash

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

set -e

# If some BUILD file breaks fuel 1.0, use BUILD.bazel instead.
find fueling -name BUILD.bazel | \
while read filename; do
  cp "${filename}" "${filename%.*}"
done

bazel build -c opt //...
