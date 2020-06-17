#!/usr/bin/env bash

# Fail on any error.
set -e

# Goto fuel root
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# For CI robot.
if [ "$1" == "--ci" ]; then
  source ./tools/docker_version.sh
  APOLLO_ROOT="/home/apollo/apollo-bazel2.x"
  docker run --privileged --rm \
      --net host \
      -v $(pwd):/fuel \
      -v ${APOLLO_ROOT}:/apollo \
      -w /fuel \
      ${IMAGE} bash /fuel/tools/check.sh
  exit $?
fi

source /usr/local/miniconda/bin/activate fuel

echo "######################### Build #########################"
bash /fuel/tools/build_local.sh

echo "######################### Test #########################"
bazel test //...

echo "######################### Lint #########################"
bash /fuel/tools/lint.sh

echo "######################### All check passed! #########################"
