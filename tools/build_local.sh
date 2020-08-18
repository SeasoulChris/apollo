#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )/.."

CACHE_DIR="/home/libs/pip-cache"
USE_CACHE="YES"

cmp -s deps/default.txt "${CACHE_DIR}/default.txt"
if [ $? -ne 0 ]; then
  echo "deps/default.txt changed since last caching. We will refresh the local pip-cache."
  sudo pip3 download -i https://mirrors.aliyun.com/pypi/simple/ -r deps/default.txt -d ${CACHE_DIR}
  if [ $? -eq 0 ]; then
    sudo cp deps/default.txt ${CACHE_DIR}/default.txt
  else
    echo "Failed to refresh pip-cache. We will use remote pypi repo as usual."
    USE_CACHE="NO"
  fi
fi

if [ "$USE_CACHE" = "YES" ]; then
  # For bazel commands, WORKSPACE.bazel has higher priority than WORKSPACE.
  sed 's|# EXTRA_PIP_ARGS|"--find-links=file:///home/libs/pip-cache/ -i https://mirrors.aliyun.com/pypi/simple/ ",|g' \
      WORKSPACE > WORKSPACE.bazel
else
  rm -f WORKSPACE.bazel
fi

env USE_CACHE="${USE_CACHE}" bash tools/build.sh $@
