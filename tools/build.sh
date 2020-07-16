#!/usr/bin/env bash

TOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

if [ -f "WORKSPACE.bazel" ]; then
  echo "###### You are building with local pip-cache! ######"
fi

DISTDIR="/fuel/.cache/distdir"
mkdir -p "${DISTDIR}"
bazel build --distdir="${DISTDIR}" ${TARGET}
