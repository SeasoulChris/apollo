#!/usr/bin/env bash

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

set -e

# If some BUILD file breaks fuel 1.0, use BUILD.v2 instead.
find fueling/ deps/ -name BUILD.v2 | \
while read filename; do
  cp "${filename}" "${filename%.*}"
done

bazel build ${TARGET}
