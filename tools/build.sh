#!/usr/bin/env bash

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

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
  bazel build --distdir="/apollo/.cache/distdir" -c opt //cyber/py_wrapper:_cyber_record_py3.so
popd

if [ -f "WORKSPACE.bazel" ]; then
  echo "###### You are building with local pip-cache! ######"
fi

bazel build --distdir="${TOP_DIR}/.cache/distdir" ${TARGET}
