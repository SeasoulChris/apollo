#!/usr/bin/env bash

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

set -e

# Build widely-used apollo modules.
pushd /apollo
  ./apollo.sh build_py
  bazel build -c opt //cyber/py_wrapper:_cyber_record_py3.so
popd

bazel build ${TARGET}
