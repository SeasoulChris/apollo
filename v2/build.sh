#!/usr/bin/env bash

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# TODO(xiaoxq): Retire after V2 launch.
grep -r '/apollo/modules/data/fuel' fueling/ | awk -F: '{print $1}' | uniq | xargs sed -i 's|/apollo/modules/data/fuel|/fuel|g'
grep -r 'modules/data/fuel' fueling/ | awk -F: '{print $1}' | uniq | xargs sed -i 's|modules/data/fuel/||g'
grep -r 'modules.data.fuel' fueling/ | awk -F: '{print $1}' | uniq | xargs sed -i 's|modules.data.fuel.||g'

set -e

# Build widely-used apollo modules.
pushd /apollo
  ./apollo.sh build_py
  bazel build -c opt //cyber/py_wrapper:_cyber_record_py3.so
popd

# If some BUILD file breaks fuel 1.0, use BUILD.v2 instead.
find fueling learning_algorithms -name BUILD.v2 | \
while read filename; do
  cp "${filename}" "${filename%.*}"
done

bazel build ${TARGET}
