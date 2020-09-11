#!/usr/bin/env bash

if [ -z "$1" ]; then
  TARGET="//..."
else
  TARGET="$@"
fi

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

if [ -f "WORKSPACE.bazel" ]; then
  if [ "$USE_CACHE" = "YES" ]; then
    echo "###### You are building with local pip-cache! ######"
  else
    rm -f "WORKSPACE.bazel"
  fi
fi

DISTDIR="/fuel/.cache/distdir"
mkdir -p "${DISTDIR}"
bazel build --distdir="${DISTDIR}" ${TARGET}
bash /fuel/tools/build_apollo.sh
