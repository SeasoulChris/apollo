#!/usr/bin/env bash

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

set -e

# Build apollo
pushd /apollo
  ./apollo.sh build_py
  bazel build \
      //cyber/py_wrapper:_cyber_record_py3.so
popd

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# TODO(xiaoxq): Retire after V2 launch.
sed -i 's|modules/data/fuel/||' fueling/autotuner/proto/*.proto

# If some BUILD file breaks fuel 1.0, use BUILD.v2 instead.
find fueling deps learning_algorithms v2 -name BUILD.v2 | \
while read filename; do
  cp "${filename}" "${filename%.*}"
done

bazel build ${TARGET}
