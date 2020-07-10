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
  bazel build --distdir="/apollo/.cache/distdir" -c opt \
      //cyber/python/cyber_py3:record \
      $( bazel query 'kind("py_library", //...)' | grep pb2$ )
popd

if [ -f "WORKSPACE.bazel" ]; then
  echo "###### You are building with local pip-cache! ######"
fi

DISTDIR="${TOP_DIR}/.cache/distdir"
mkdir -p "${DISTDIR}"
bazel build --distdir="${DISTDIR}" ${TARGET}
